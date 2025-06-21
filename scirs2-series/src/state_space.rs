//! State-space models for time series analysis
//!
//! Implements structural time series models, dynamic linear models, and Kalman filtering

use ndarray::{Array1, Array2, ArrayBase, Data, Ix1, ScalarOperand};
use num_traits::{Float, FromPrimitive};
use std::fmt::{Debug, Display};

use crate::error::{Result, TimeSeriesError};

/// State vector representation for state-space models
#[derive(Debug, Clone)]
pub struct StateVector<F> {
    /// State values
    pub state: Array1<F>,
    /// Covariance matrix
    pub covariance: Array2<F>,
}

/// Result of Kalman smoothing containing all estimates
#[derive(Debug, Clone)]
pub struct SmootherResult<F> {
    /// Smoothed state estimates (backward pass)
    pub smoothed_states: Vec<StateVector<F>>,
    /// Filtered state estimates (forward pass)
    pub filtered_states: Vec<StateVector<F>>,
    /// Predicted state estimates (one-step ahead)
    pub predicted_states: Vec<StateVector<F>>,
    /// Log-likelihood of the observations
    pub log_likelihood: F,
}

/// Parameters for EM algorithm
#[derive(Debug, Clone)]
pub struct EMParameters<F> {
    /// Maximum number of EM iterations
    pub max_iterations: usize,
    /// Convergence tolerance for log-likelihood
    pub tolerance: F,
    /// Whether to print convergence information
    pub verbose: bool,
}

impl<F> Default for EMParameters<F>
where
    F: Float + FromPrimitive,
{
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: F::from(1e-6).unwrap(),
            verbose: false,
        }
    }
}

/// Result of EM algorithm
#[derive(Debug, Clone)]
pub struct EMResult<F> {
    /// Final log-likelihood
    pub log_likelihood: F,
    /// Number of iterations until convergence
    pub iterations: usize,
    /// Whether the algorithm converged
    pub converged: bool,
    /// History of log-likelihood values
    pub log_likelihood_history: Vec<F>,
}

/// Observation model for state-space models
#[derive(Debug, Clone)]
pub struct ObservationModel<F> {
    /// Observation matrix (maps state to observations)
    pub observation_matrix: Array2<F>,
    /// Observation noise covariance
    pub observation_noise: Array2<F>,
}

/// State transition model
#[derive(Debug, Clone)]
pub struct StateTransition<F> {
    /// State transition matrix
    pub transition_matrix: Array2<F>,
    /// Process noise covariance
    pub process_noise: Array2<F>,
}

/// Kalman filter for state estimation
#[derive(Debug, Clone)]
pub struct KalmanFilter<F> {
    /// Current state estimate
    pub state: StateVector<F>,
    /// State transition model
    pub transition: StateTransition<F>,
    /// Observation model
    pub observation: ObservationModel<F>,
}

impl<F> KalmanFilter<F>
where
    F: Float + FromPrimitive + Debug + Display + ScalarOperand,
{
    /// Create a new Kalman filter
    pub fn new(
        initial_state: StateVector<F>,
        transition: StateTransition<F>,
        observation: ObservationModel<F>,
    ) -> Self {
        Self {
            state: initial_state,
            transition,
            observation,
        }
    }

    /// Predict step (time update)
    pub fn predict(&mut self) -> Result<()> {
        // State prediction: x_k|k-1 = F * x_k-1|k-1
        let predicted_state = self.transition.transition_matrix.dot(&self.state.state);

        // Covariance prediction: P_k|k-1 = F * P_k-1|k-1 * F^T + Q
        let temp = self
            .transition
            .transition_matrix
            .dot(&self.state.covariance);
        let predicted_covariance =
            temp.dot(&self.transition.transition_matrix.t()) + &self.transition.process_noise;

        self.state.state = predicted_state;
        self.state.covariance = predicted_covariance;

        Ok(())
    }

    /// Update step (measurement update)
    pub fn update(&mut self, observation: &Array1<F>) -> Result<()> {
        // Innovation: y_k = z_k - H * x_k|k-1
        let predicted_observation = self.observation.observation_matrix.dot(&self.state.state);
        let innovation = observation - &predicted_observation;

        // Innovation covariance: S_k = H * P_k|k-1 * H^T + R
        let temp = self
            .observation
            .observation_matrix
            .dot(&self.state.covariance);
        let innovation_covariance = temp.dot(&self.observation.observation_matrix.t())
            + &self.observation.observation_noise;

        // Kalman gain: K_k = P_k|k-1 * H^T * S_k^-1
        let kalman_gain = compute_kalman_gain(
            &self.state.covariance,
            &self.observation.observation_matrix,
            &innovation_covariance,
        )?;

        // State update: x_k|k = x_k|k-1 + K_k * y_k
        let state_update = kalman_gain.dot(&innovation);
        self.state.state = &self.state.state + &state_update;

        // Covariance update: P_k|k = (I - K_k * H) * P_k|k-1
        let n = self.state.state.len();
        let identity = Array2::eye(n);
        let temp = kalman_gain.dot(&self.observation.observation_matrix);
        let covariance_factor = identity - temp;
        self.state.covariance = covariance_factor.dot(&self.state.covariance);

        Ok(())
    }

    /// Run the filter on a time series
    pub fn filter(&mut self, observations: &Array1<F>) -> Result<Vec<StateVector<F>>> {
        let mut states = Vec::with_capacity(observations.len());

        for obs in observations.iter() {
            self.predict()?;
            self.update(&Array1::from_elem(1, *obs))?;
            states.push(self.state.clone());
        }

        Ok(states)
    }

    /// Run the forward-backward smoother on a time series (Rauch-Tung-Striebel smoother)
    pub fn smooth(&mut self, observations: &Array1<F>) -> Result<SmootherResult<F>> {
        let n = observations.len();

        // Forward pass - store predicted and filtered states
        let mut predicted_states = Vec::with_capacity(n);
        let mut filtered_states = Vec::with_capacity(n);

        for obs in observations.iter() {
            // Store predicted state before update
            self.predict()?;
            predicted_states.push(self.state.clone());

            // Update and store filtered state
            self.update(&Array1::from_elem(1, *obs))?;
            filtered_states.push(self.state.clone());
        }

        // Backward pass - smooth the estimates
        let mut smoothed_states = vec![filtered_states[n - 1].clone()];

        for t in (0..(n - 1)).rev() {
            let smoothed_next = &smoothed_states[0];
            let filtered_current = &filtered_states[t];
            let predicted_next = &predicted_states[t + 1];

            // Smoother gain: A_t = P_{t|t} * F^T * P_{t+1|t}^{-1}
            let temp = filtered_current
                .covariance
                .dot(&self.transition.transition_matrix.t());
            let gain = temp.dot(&pseudo_inverse(&predicted_next.covariance)?);

            // Smoothed state: x_{t|T} = x_{t|t} + A_t * (x_{t+1|T} - x_{t+1|t})
            let state_diff = &smoothed_next.state - &predicted_next.state;
            let smoothed_state = &filtered_current.state + &gain.dot(&state_diff);

            // Smoothed covariance: P_{t|T} = P_{t|t} + A_t * (P_{t+1|T} - P_{t+1|t}) * A_t^T
            let cov_diff = &smoothed_next.covariance - &predicted_next.covariance;
            let temp = gain.dot(&cov_diff);
            let smoothed_covariance = &filtered_current.covariance + &temp.dot(&gain.t());

            smoothed_states.insert(
                0,
                StateVector {
                    state: smoothed_state,
                    covariance: smoothed_covariance,
                },
            );
        }

        Ok(SmootherResult {
            smoothed_states,
            filtered_states,
            predicted_states,
            log_likelihood: self.compute_log_likelihood(observations)?,
        })
    }

    /// Compute log-likelihood of observations given current parameters
    pub fn compute_log_likelihood(&mut self, observations: &Array1<F>) -> Result<F> {
        let mut log_likelihood = F::zero();
        let _n = observations.len();

        // Reset filter to initial state
        let initial_state = self.state.clone();

        for obs in observations.iter() {
            self.predict()?;

            // Compute innovation and its covariance
            let predicted_obs = self.observation.observation_matrix.dot(&self.state.state);
            let innovation = Array1::from_elem(1, *obs) - &predicted_obs;

            let temp = self
                .observation
                .observation_matrix
                .dot(&self.state.covariance);
            let innovation_cov = temp.dot(&self.observation.observation_matrix.t())
                + &self.observation.observation_noise;

            // Add to log-likelihood: -0.5 * (log(2π) + log|S| + y'S^{-1}y)
            let log_det = matrix_log_determinant(&innovation_cov);
            let inv_innovation_cov = matrix_inverse(&innovation_cov)?;
            let quadratic_form = innovation.dot(&inv_innovation_cov.dot(&innovation));

            let dim = F::from(innovation_cov.nrows()).unwrap();
            let two_pi = F::from(2.0 * std::f64::consts::PI).unwrap();

            log_likelihood = log_likelihood
                - F::from(0.5).unwrap() * (dim * two_pi.ln() + log_det + quadratic_form);

            self.update(&Array1::from_elem(1, *obs))?;
        }

        // Restore initial state
        self.state = initial_state;

        Ok(log_likelihood)
    }
}

/// Compute Kalman gain
fn compute_kalman_gain<F>(
    covariance: &Array2<F>,
    observation_matrix: &Array2<F>,
    innovation_covariance: &Array2<F>,
) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Display + ScalarOperand,
{
    // K = P * H^T * S^-1
    let temp = covariance.dot(&observation_matrix.t());

    // Use pseudo-inverse for numerical stability
    let inv_innovation = pseudo_inverse(innovation_covariance)?;
    Ok(temp.dot(&inv_innovation))
}

/// Matrix inversion using LU decomposition with partial pivoting
fn matrix_inverse<F>(matrix: &Array2<F>) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Display + ScalarOperand,
{
    let (m, n) = matrix.dim();
    if m != n {
        return Err(TimeSeriesError::InvalidInput(
            "Matrix must be square for inversion".to_string(),
        ));
    }

    if n == 0 {
        return Ok(Array2::zeros((0, 0)));
    }

    // Check for diagonal matrix (optimization)
    let mut is_diagonal = true;
    for i in 0..m {
        for j in 0..n {
            if i != j && matrix[[i, j]].abs() > F::from(1e-12).unwrap() {
                is_diagonal = false;
                break;
            }
        }
        if !is_diagonal {
            break;
        }
    }

    if is_diagonal {
        // For diagonal matrices, invert the diagonal elements
        let mut inv = Array2::zeros((m, n));
        for i in 0..m {
            let diag = matrix[[i, i]];
            if diag.abs() < F::from(1e-12).unwrap() {
                return Err(TimeSeriesError::NumericalInstability(
                    "Singular matrix: zero diagonal element".to_string(),
                ));
            }
            inv[[i, i]] = F::one() / diag;
        }
        return Ok(inv);
    }

    // General case: LU decomposition with partial pivoting
    let mut lu = matrix.clone();
    let mut perm = (0..n).collect::<Vec<_>>();
    let mut identity = Array2::eye(n);

    // LU decomposition with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = lu[[col, col]].abs();
        let mut max_row = col;

        for row in (col + 1)..n {
            let val = lu[[row, col]].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        // Swap rows if needed
        if max_row != col {
            for j in 0..n {
                let temp = lu[[col, j]];
                lu[[col, j]] = lu[[max_row, j]];
                lu[[max_row, j]] = temp;

                let temp = identity[[col, j]];
                identity[[col, j]] = identity[[max_row, j]];
                identity[[max_row, j]] = temp;
            }
            perm.swap(col, max_row);
        }

        // Check for near-zero pivot
        if lu[[col, col]].abs() < F::from(1e-12).unwrap() {
            return Err(TimeSeriesError::NumericalInstability(
                "Singular matrix: near-zero pivot".to_string(),
            ));
        }

        // Eliminate below pivot
        for row in (col + 1)..n {
            let factor = lu[[row, col]] / lu[[col, col]];
            lu[[row, col]] = factor; // Store multiplier

            for j in (col + 1)..n {
                lu[[row, j]] = lu[[row, j]] - factor * lu[[col, j]];
            }

            for j in 0..n {
                identity[[row, j]] = identity[[row, j]] - factor * identity[[col, j]];
            }
        }
    }

    // Back substitution for each column of the identity matrix
    let mut inverse = Array2::zeros((n, n));
    for col in 0..n {
        let mut x = vec![F::zero(); n];
        for i in 0..n {
            x[i] = identity[[i, col]];
        }

        // Solve Ux = y
        for i in (0..n).rev() {
            let mut sum = F::zero();
            for j in (i + 1)..n {
                sum = sum + lu[[i, j]] * x[j];
            }
            x[i] = (x[i] - sum) / lu[[i, i]];
        }

        for i in 0..n {
            inverse[[i, col]] = x[i];
        }
    }

    Ok(inverse)
}

/// Pseudo-inverse implementation using the matrix inverse with regularization
fn pseudo_inverse<F>(matrix: &Array2<F>) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Display + ScalarOperand,
{
    let (m, n) = matrix.dim();

    if m == n {
        // Square matrix - try direct inversion
        match matrix_inverse(matrix) {
            Ok(inv) => return Ok(inv),
            Err(_) => {
                // Fall through to regularized approach
            }
        }
    }

    // For non-square or singular matrices, use (A'A + λI)^(-1) A'
    let regularization = F::from(1e-8).unwrap();
    let at = matrix.t();
    let ata = at.dot(matrix);

    // Add regularization to diagonal
    let mut regularized = ata.clone();
    for i in 0..ata.nrows().min(ata.ncols()) {
        regularized[[i, i]] = regularized[[i, i]] + regularization;
    }

    let inv_regularized = matrix_inverse(&regularized)?;
    Ok(inv_regularized.dot(&at))
}

/// Calculate outer product of two vectors
fn outer_product<F>(a: &Array1<F>, b: &Array1<F>) -> Array2<F>
where
    F: Float + Clone,
{
    let mut result = Array2::zeros((a.len(), b.len()));
    for i in 0..a.len() {
        for j in 0..b.len() {
            result[[i, j]] = a[i] * b[j];
        }
    }
    result
}

/// Calculate log determinant of a matrix using LU decomposition
fn matrix_log_determinant<F>(matrix: &Array2<F>) -> F
where
    F: Float + FromPrimitive + Debug + Display + ScalarOperand,
{
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return F::neg_infinity(); // Invalid matrix
    }

    if n == 0 {
        return F::zero();
    }

    // Create working copy for LU decomposition
    let mut lu = matrix.clone();
    let mut sign = F::one();

    // LU decomposition with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = lu[[col, col]].abs();
        let mut max_row = col;

        for row in (col + 1)..n {
            let val = lu[[row, col]].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        // Swap rows if needed
        if max_row != col {
            for j in col..n {
                let temp = lu[[col, j]];
                lu[[col, j]] = lu[[max_row, j]];
                lu[[max_row, j]] = temp;
            }
            sign = -sign; // Row swap changes determinant sign
        }

        // Check for zero pivot (singular matrix)
        if lu[[col, col]].abs() < F::from(1e-12).unwrap() {
            return F::neg_infinity(); // log(0) = -infinity
        }

        // Eliminate below pivot
        for row in (col + 1)..n {
            let factor = lu[[row, col]] / lu[[col, col]];

            for j in (col + 1)..n {
                lu[[row, j]] = lu[[row, j]] - factor * lu[[col, j]];
            }
        }
    }

    // Calculate log determinant from diagonal elements
    let mut log_det = F::zero();
    for i in 0..n {
        let diag_element = lu[[i, i]];
        if diag_element.abs() < F::from(1e-12).unwrap() {
            return F::neg_infinity(); // Singular matrix
        }
        log_det = log_det + diag_element.abs().ln();
    }

    // Account for sign
    if sign < F::zero() {
        // For negative determinant, we return ln(|det|)
        log_det
    } else {
        log_det
    }
}

/// Structural time series model
#[derive(Debug, Clone)]
pub struct StructuralModel<F> {
    /// Level component
    pub level: Option<StateTransition<F>>,
    /// Trend component
    pub trend: Option<StateTransition<F>>,
    /// Seasonal component
    pub seasonal: Option<StateTransition<F>>,
    /// Observation model
    pub observation: ObservationModel<F>,
}

impl<F> StructuralModel<F>
where
    F: Float + FromPrimitive + Debug + Display + ScalarOperand,
{
    /// Create a local level model (random walk plus noise)
    pub fn local_level(sigma_level: F, sigma_obs: F) -> Result<Self> {
        let transition = StateTransition {
            transition_matrix: Array2::eye(1),
            process_noise: Array2::from_elem((1, 1), sigma_level * sigma_level),
        };

        let observation = ObservationModel {
            observation_matrix: Array2::eye(1),
            observation_noise: Array2::from_elem((1, 1), sigma_obs * sigma_obs),
        };

        Ok(Self {
            level: Some(transition),
            trend: None,
            seasonal: None,
            observation,
        })
    }

    /// Create a local linear trend model
    pub fn local_linear_trend(sigma_level: F, sigma_trend: F, sigma_obs: F) -> Result<Self> {
        // State: [level, trend]
        let transition_matrix = Array2::from_shape_vec(
            (2, 2),
            vec![F::one(), F::one(), F::zero(), F::one()],
        )
        .map_err(|e| {
            TimeSeriesError::InvalidInput(format!("Failed to create transition matrix: {}", e))
        })?;

        let process_noise = Array2::from_shape_vec(
            (2, 2),
            vec![
                sigma_level * sigma_level,
                F::zero(),
                F::zero(),
                sigma_trend * sigma_trend,
            ],
        )
        .map_err(|e| {
            TimeSeriesError::InvalidInput(format!("Failed to create process noise matrix: {}", e))
        })?;

        let transition = StateTransition {
            transition_matrix,
            process_noise,
        };

        let observation = ObservationModel {
            observation_matrix: Array2::from_shape_vec((1, 2), vec![F::one(), F::zero()]).map_err(
                |e| {
                    TimeSeriesError::InvalidInput(format!(
                        "Failed to create observation matrix: {}",
                        e
                    ))
                },
            )?,
            observation_noise: Array2::from_elem((1, 1), sigma_obs * sigma_obs),
        };

        Ok(Self {
            level: Some(transition),
            trend: None,
            seasonal: None,
            observation,
        })
    }

    /// Fit the model to data using maximum likelihood estimation with EM algorithm
    pub fn fit<S>(&mut self, data: &ArrayBase<S, Ix1>) -> Result<EMResult<F>>
    where
        S: Data<Elem = F>,
    {
        scirs2_core::validation::check_array_finite(data, "data")?;

        let em_params = EMParameters::default();
        self.fit_with_em(data, &em_params)
    }

    /// Fit the model using EM algorithm with custom parameters
    pub fn fit_with_em<S>(
        &mut self,
        data: &ArrayBase<S, Ix1>,
        em_params: &EMParameters<F>,
    ) -> Result<EMResult<F>>
    where
        S: Data<Elem = F>,
    {
        let _n = data.len();
        let observations = data.to_owned();

        let mut log_likelihood_history = Vec::new();
        let mut current_log_likelihood = F::neg_infinity();
        let mut converged = false;

        for iteration in 0..em_params.max_iterations {
            // E-step: Run Kalman filter and smoother
            let mut kf = self.create_kalman_filter()?;
            let smoother_result = kf.smooth(&observations)?;

            let new_log_likelihood = smoother_result.log_likelihood;
            log_likelihood_history.push(new_log_likelihood);

            if em_params.verbose {
                println!(
                    "EM Iteration {}: Log-likelihood = {:.6}",
                    iteration, new_log_likelihood
                );
            }

            // Check convergence
            if iteration > 0 {
                let improvement = new_log_likelihood - current_log_likelihood;
                if improvement.abs() < em_params.tolerance {
                    converged = true;
                    if em_params.verbose {
                        println!("EM converged after {} iterations", iteration);
                    }
                    break;
                }

                if improvement < F::zero() && em_params.verbose {
                    println!(
                        "Warning: Log-likelihood decreased in EM iteration {}",
                        iteration
                    );
                }
            }

            current_log_likelihood = new_log_likelihood;

            // M-step: Update parameters based on sufficient statistics
            self.update_parameters(&smoother_result, &observations)?;
        }

        Ok(EMResult {
            log_likelihood: current_log_likelihood,
            iterations: log_likelihood_history.len(),
            converged,
            log_likelihood_history,
        })
    }

    /// Create a Kalman filter from the structural model
    fn create_kalman_filter(&self) -> Result<KalmanFilter<F>> {
        // For structural models, we need to combine level, trend, and seasonal components
        let state_dim = self.get_state_dimension();

        // Create initial state
        let initial_state = StateVector {
            state: Array1::zeros(state_dim),
            covariance: Array2::eye(state_dim) * F::from(100.0).unwrap(), // Diffuse prior
        };

        // Create combined transition and observation matrices
        let transition = self.create_combined_transition_matrix()?;
        let observation = self.observation.clone();

        Ok(KalmanFilter::new(initial_state, transition, observation))
    }

    /// Get the total state dimension (level + trend + seasonal)
    fn get_state_dimension(&self) -> usize {
        let mut dim = 0;

        if let Some(level) = &self.level {
            dim += level.transition_matrix.nrows();
        }

        if let Some(trend) = &self.trend {
            dim += trend.transition_matrix.nrows();
        }

        if let Some(seasonal) = &self.seasonal {
            dim += seasonal.transition_matrix.nrows();
        }

        dim.max(1) // At least one state
    }

    /// Create combined transition matrix from components
    fn create_combined_transition_matrix(&self) -> Result<StateTransition<F>> {
        let state_dim = self.get_state_dimension();
        let mut transition_matrix = Array2::zeros((state_dim, state_dim));
        let mut process_noise = Array2::zeros((state_dim, state_dim));

        let mut current_idx = 0;

        // Add level component
        if let Some(level) = &self.level {
            let level_dim = level.transition_matrix.nrows();
            for i in 0..level_dim {
                for j in 0..level_dim {
                    transition_matrix[[current_idx + i, current_idx + j]] =
                        level.transition_matrix[[i, j]];
                    process_noise[[current_idx + i, current_idx + j]] = level.process_noise[[i, j]];
                }
            }
            current_idx += level_dim;
        }

        // Add trend component
        if let Some(trend) = &self.trend {
            let trend_dim = trend.transition_matrix.nrows();
            for i in 0..trend_dim {
                for j in 0..trend_dim {
                    transition_matrix[[current_idx + i, current_idx + j]] =
                        trend.transition_matrix[[i, j]];
                    process_noise[[current_idx + i, current_idx + j]] = trend.process_noise[[i, j]];
                }
            }
            current_idx += trend_dim;
        }

        // Add seasonal component
        if let Some(seasonal) = &self.seasonal {
            let seasonal_dim = seasonal.transition_matrix.nrows();
            for i in 0..seasonal_dim {
                for j in 0..seasonal_dim {
                    transition_matrix[[current_idx + i, current_idx + j]] =
                        seasonal.transition_matrix[[i, j]];
                    process_noise[[current_idx + i, current_idx + j]] =
                        seasonal.process_noise[[i, j]];
                }
            }
        }

        Ok(StateTransition {
            transition_matrix,
            process_noise,
        })
    }

    /// Update model parameters based on sufficient statistics from E-step
    fn update_parameters(
        &mut self,
        smoother_result: &SmootherResult<F>,
        observations: &Array1<F>,
    ) -> Result<()> {
        let n = observations.len();

        // Compute sufficient statistics
        let mut sum_x: Array1<F> = Array1::zeros(self.get_state_dimension());
        let mut sum_xx: Array2<F> =
            Array2::zeros((self.get_state_dimension(), self.get_state_dimension()));
        let mut sum_x_lag: Array2<F> =
            Array2::zeros((self.get_state_dimension(), self.get_state_dimension()));
        let mut sum_y = F::zero();
        let mut sum_yy = F::zero();
        let mut sum_xy: Array1<F> = Array1::zeros(self.get_state_dimension());

        // Accumulate sufficient statistics
        for t in 0..n {
            let state = &smoother_result.smoothed_states[t];

            sum_x = sum_x + &state.state;
            sum_xx = sum_xx + &state.covariance + &outer_product(&state.state, &state.state);
            sum_y = sum_y + observations[t];
            sum_yy = sum_yy + observations[t] * observations[t];
            sum_xy = sum_xy + &state.state * observations[t];

            if t > 0 {
                let prev_state = &smoother_result.smoothed_states[t - 1];
                sum_x_lag = sum_x_lag + &outer_product(&state.state, &prev_state.state);
            }
        }

        // Update observation noise variance (simplified for univariate case)
        let n_f = F::from(n).unwrap();
        let term1: F = F::from(2.0).unwrap() * sum_xy[0] * sum_y / n_f;
        let term2: F = sum_xx[[0, 0]] * sum_y * sum_y / (n_f * n_f);
        let residual_variance = (sum_yy - term1 + term2) / n_f;

        // Ensure positive variance
        if residual_variance > F::zero() {
            self.observation.observation_noise[[0, 0]] = residual_variance;
        }

        // Update process noise variances (simplified approach)
        if let Some(level) = &mut self.level {
            let level_variance: F = sum_xx[[0, 0]] / n_f - (sum_x[0] / n_f) * (sum_x[0] / n_f);
            if level_variance > F::zero() {
                level.process_noise[[0, 0]] = level_variance * F::from(0.1).unwrap();
                // Scale down for stability
            }
        }

        Ok(())
    }

    /// Forecast using the fitted model
    pub fn forecast(&self, steps: usize) -> Result<Array1<F>> {
        if steps == 0 {
            return Ok(Array1::zeros(0));
        }

        let mut kf = self.create_kalman_filter()?;
        let mut forecasts = Array1::zeros(steps);

        // Generate forecasts by predicting forward
        for t in 0..steps {
            kf.predict()?;

            // Extract observation from state
            let forecast = kf.observation.observation_matrix.dot(&kf.state.state);
            forecasts[t] = forecast[0]; // Assuming univariate observations
        }

        Ok(forecasts)
    }

    /// Create a seasonal state-space model
    pub fn seasonal(
        sigma_level: F,
        sigma_seasonal: F,
        sigma_obs: F,
        period: usize,
    ) -> Result<Self> {
        if period < 2 {
            return Err(TimeSeriesError::InvalidInput(
                "Seasonal period must be at least 2".to_string(),
            ));
        }

        // Level component (random walk)
        let level_transition = StateTransition {
            transition_matrix: Array2::eye(1),
            process_noise: Array2::from_elem((1, 1), sigma_level * sigma_level),
        };

        // Seasonal component (sum-to-zero seasonal states)
        let mut seasonal_matrix = Array2::zeros((period - 1, period - 1));

        // First row: [-1, -1, ..., -1]
        for j in 0..(period - 1) {
            seasonal_matrix[[0, j]] = -F::one();
        }

        // Identity for other rows (shifting states)
        for i in 1..(period - 1) {
            seasonal_matrix[[i, i - 1]] = F::one();
        }

        let seasonal_transition = StateTransition {
            transition_matrix: seasonal_matrix,
            process_noise: {
                let mut noise = Array2::zeros((period - 1, period - 1));
                noise[[0, 0]] = sigma_seasonal * sigma_seasonal;
                noise
            },
        };

        // Observation model: observe level + first seasonal state
        let obs_dim = 1 + (period - 1); // level + seasonal states
        let mut obs_matrix = Array2::zeros((1, obs_dim));
        obs_matrix[[0, 0]] = F::one(); // level
        obs_matrix[[0, 1]] = F::one(); // first seasonal state

        let observation = ObservationModel {
            observation_matrix: obs_matrix,
            observation_noise: Array2::from_elem((1, 1), sigma_obs * sigma_obs),
        };

        Ok(Self {
            level: Some(level_transition),
            trend: None,
            seasonal: Some(seasonal_transition),
            observation,
        })
    }
}

/// Unobserved components model
#[derive(Debug, Clone)]
pub struct UnobservedComponentsModel<F> {
    /// Structural model
    pub model: StructuralModel<F>,
    /// Exogenous variables (optional)
    pub exogenous: Option<Array2<F>>,
}

/// Dynamic linear model
#[derive(Debug, Clone)]
pub struct DynamicLinearModel<F> {
    /// State transition
    pub transition: StateTransition<F>,
    /// Observation model
    pub observation: ObservationModel<F>,
    /// Control input (optional)
    pub control: Option<Array2<F>>,
    /// Control matrix (optional)
    pub control_matrix: Option<Array2<F>>,
}

impl<F> DynamicLinearModel<F>
where
    F: Float + FromPrimitive + Debug + Display + ScalarOperand,
{
    /// Create a new dynamic linear model
    pub fn new(transition: StateTransition<F>, observation: ObservationModel<F>) -> Self {
        Self {
            transition,
            observation,
            control: None,
            control_matrix: None,
        }
    }

    /// Add control input
    pub fn with_control(mut self, control: Array2<F>, control_matrix: Array2<F>) -> Self {
        self.control = Some(control);
        self.control_matrix = Some(control_matrix);
        self
    }

    /// Fit the model using EM algorithm
    pub fn fit<S>(&mut self, data: &ArrayBase<S, Ix1>) -> Result<EMResult<F>>
    where
        S: Data<Elem = F>,
    {
        scirs2_core::validation::check_array_finite(data, "data")?;

        let em_params = EMParameters::default();
        self.fit_with_em(data, &em_params)
    }

    /// Fit the model using EM algorithm with custom parameters
    pub fn fit_with_em<S>(
        &mut self,
        data: &ArrayBase<S, Ix1>,
        em_params: &EMParameters<F>,
    ) -> Result<EMResult<F>>
    where
        S: Data<Elem = F>,
    {
        let observations = data.to_owned();

        let mut log_likelihood_history = Vec::new();
        let mut current_log_likelihood = F::neg_infinity();
        let mut converged = false;

        for iteration in 0..em_params.max_iterations {
            // E-step: Run Kalman filter and smoother
            let smoother_result = self.smooth(&observations)?;

            let new_log_likelihood = smoother_result.log_likelihood;
            log_likelihood_history.push(new_log_likelihood);

            if em_params.verbose {
                println!(
                    "DLM EM Iteration {}: Log-likelihood = {:.6}",
                    iteration, new_log_likelihood
                );
            }

            // Check convergence
            if iteration > 0 {
                let improvement = new_log_likelihood - current_log_likelihood;
                if improvement.abs() < em_params.tolerance {
                    converged = true;
                    if em_params.verbose {
                        println!("DLM EM converged after {} iterations", iteration);
                    }
                    break;
                }

                if improvement < F::zero() && em_params.verbose {
                    println!(
                        "Warning: Log-likelihood decreased in DLM EM iteration {}",
                        iteration
                    );
                }
            }

            current_log_likelihood = new_log_likelihood;

            // M-step: Update parameters
            self.update_dlm_parameters(&smoother_result, &observations)?;
        }

        Ok(EMResult {
            log_likelihood: current_log_likelihood,
            iterations: log_likelihood_history.len(),
            converged,
            log_likelihood_history,
        })
    }

    /// Smooth the state estimates using Rauch-Tung-Striebel smoother
    pub fn smooth(&self, observations: &Array1<F>) -> Result<SmootherResult<F>> {
        // Create initial state for the filter
        let state_dim = self.transition.transition_matrix.nrows();
        let initial_state = StateVector {
            state: Array1::zeros(state_dim),
            covariance: Array2::eye(state_dim) * F::from(100.0).unwrap(), // Diffuse prior
        };

        let mut kf = KalmanFilter::new(
            initial_state,
            self.transition.clone(),
            self.observation.clone(),
        );

        kf.smooth(observations)
    }

    /// Update DLM parameters based on sufficient statistics from E-step
    fn update_dlm_parameters(
        &mut self,
        smoother_result: &SmootherResult<F>,
        observations: &Array1<F>,
    ) -> Result<()> {
        let n = observations.len();
        let state_dim = self.transition.transition_matrix.nrows();

        // Compute sufficient statistics
        let mut sum_x: Array1<F> = Array1::zeros(state_dim);
        let mut sum_xx: Array2<F> = Array2::zeros((state_dim, state_dim));
        let mut sum_x_lag: Array2<F> = Array2::zeros((state_dim, state_dim));
        let mut sum_xx_lag: Array2<F> = Array2::zeros((state_dim, state_dim));
        let mut sum_y = F::zero();
        let mut sum_yy = F::zero();
        let mut sum_xy: Array1<F> = Array1::zeros(state_dim);

        // Accumulate sufficient statistics
        for t in 0..n {
            let state = &smoother_result.smoothed_states[t];

            sum_x = sum_x + &state.state;
            sum_xx = sum_xx + &state.covariance + &outer_product(&state.state, &state.state);
            sum_y = sum_y + observations[t];
            sum_yy = sum_yy + observations[t] * observations[t];
            sum_xy = sum_xy + &state.state * observations[t];

            if t > 0 {
                let prev_state = &smoother_result.smoothed_states[t - 1];
                sum_x_lag = sum_x_lag + &outer_product(&state.state, &prev_state.state);
                sum_xx_lag = sum_xx_lag
                    + &prev_state.covariance
                    + &outer_product(&prev_state.state, &prev_state.state);
            }
        }

        // Update observation matrix (simplified for univariate observations)
        if sum_xx[[0, 0]] > F::epsilon() {
            let new_obs_coeff = sum_xy[0] / sum_xx[[0, 0]];
            self.observation.observation_matrix[[0, 0]] = new_obs_coeff;
        }

        // Update observation noise variance
        let n_f = F::from(n).unwrap();
        let predicted_sum = self.observation.observation_matrix.row(0).dot(&sum_xy);
        let residual_variance = (sum_yy - predicted_sum) / n_f;

        if residual_variance > F::zero() {
            self.observation.observation_noise[[0, 0]] = residual_variance;
        }

        // Update transition matrix (for autoregressive models)
        if n > 1 {
            let n_transitions = F::from(n - 1).unwrap();

            // Simplified AR(1) parameter update
            if sum_xx_lag[[0, 0]] > F::epsilon() {
                let ar_coeff = sum_x_lag[[0, 0]] / sum_xx_lag[[0, 0]];
                self.transition.transition_matrix[[0, 0]] = ar_coeff;
            }

            // Update process noise
            let predicted_transitions =
                self.transition.transition_matrix[[0, 0]] * sum_x_lag[[0, 0]];
            let process_residual_variance =
                (sum_xx[[0, 0]] - predicted_transitions) / n_transitions;

            if process_residual_variance > F::zero() {
                self.transition.process_noise[[0, 0]] = process_residual_variance;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq; // Currently unused but needed for future tests
    use ndarray::array;

    #[test]
    fn test_kalman_filter_prediction() {
        let initial_state = StateVector {
            state: array![0.0],
            covariance: array![[1.0]],
        };

        let transition = StateTransition {
            transition_matrix: array![[1.0]],
            process_noise: array![[0.1]],
        };

        let observation = ObservationModel {
            observation_matrix: array![[1.0]],
            observation_noise: array![[0.1]],
        };

        let mut kf = KalmanFilter::new(initial_state, transition, observation);
        kf.predict().unwrap();

        // After prediction, covariance should increase
        assert!(kf.state.covariance[[0, 0]] > 1.0);
    }

    #[test]
    fn test_local_level_model() {
        let model = StructuralModel::local_level(0.1_f64, 0.2_f64).unwrap();
        assert!(model.level.is_some());
        assert!(model.trend.is_none());
        assert!(model.seasonal.is_none());
    }

    #[test]
    fn test_local_linear_trend_model() {
        let model = StructuralModel::local_linear_trend(0.1_f64, 0.05_f64, 0.2_f64).unwrap();
        assert!(model.level.is_some());

        let transition = model.level.unwrap();
        assert_eq!(transition.transition_matrix.dim(), (2, 2));
    }

    #[test]
    fn test_kalman_filter_update() {
        let initial_state = StateVector {
            state: array![0.0],
            covariance: array![[1.0]],
        };

        let transition = StateTransition {
            transition_matrix: array![[1.0]],
            process_noise: array![[0.1]],
        };

        let observation = ObservationModel {
            observation_matrix: array![[1.0]],
            observation_noise: array![[0.1]],
        };

        let mut kf = KalmanFilter::new(initial_state, transition, observation);

        // Make an observation
        let obs = array![1.0];
        kf.update(&obs).unwrap();

        // State should move towards observation
        assert!(kf.state.state[0] > 0.0);
        // Uncertainty should decrease after update
        assert!(kf.state.covariance[[0, 0]] < 1.0);
    }

    #[test]
    fn test_kalman_filter_sequence() {
        let initial_state = StateVector {
            state: array![0.0],
            covariance: array![[1.0]],
        };

        let transition = StateTransition {
            transition_matrix: array![[1.0]],
            process_noise: array![[0.1]],
        };

        let observation = ObservationModel {
            observation_matrix: array![[1.0]],
            observation_noise: array![[0.1]],
        };

        let mut kf = KalmanFilter::new(initial_state, transition, observation);
        let observations = array![1.0, 1.1, 0.9, 1.0];

        let states = kf.filter(&observations).unwrap();
        assert_eq!(states.len(), observations.len());

        // Final state should be near the observations
        let final_state = &states.last().unwrap().state;
        assert_relative_eq!(final_state[0], 1.0, epsilon = 0.5);
    }

    #[test]
    fn test_kalman_smoother() {
        let initial_state = StateVector {
            state: array![0.0],
            covariance: array![[1.0]],
        };

        let transition = StateTransition {
            transition_matrix: array![[0.9]], // AR(1) with coefficient 0.9
            process_noise: array![[0.1]],
        };

        let observation = ObservationModel {
            observation_matrix: array![[1.0]],
            observation_noise: array![[0.1]],
        };

        let mut kf = KalmanFilter::new(initial_state, transition, observation);
        let observations = array![1.0, 0.9, 0.8, 0.7];

        let smoother_result = kf.smooth(&observations).unwrap();

        assert_eq!(smoother_result.smoothed_states.len(), observations.len());
        assert_eq!(smoother_result.filtered_states.len(), observations.len());
        assert_eq!(smoother_result.predicted_states.len(), observations.len());

        // Smoothed estimates should be more accurate than filtered estimates
        // (This is a qualitative check - in practice we'd need more sophisticated validation)
        assert!(smoother_result.log_likelihood > f64::NEG_INFINITY);
    }

    #[test]
    fn test_structural_model_em() {
        // Create a simple local level model
        let mut model = StructuralModel::local_level(0.1_f64, 0.2_f64).unwrap();

        // Generate some test data (random walk with noise)
        let n = 50;
        let mut data = Array1::zeros(n);
        data[0] = 0.0;
        for i in 1..n {
            data[i] = data[i - 1] + 0.1 * (i as f64 / 10.0).sin() + 0.05;
        }

        // Fit the model using EM
        let em_result = model.fit(&data).unwrap();

        // Check that EM completed
        assert!(em_result.iterations > 0);
        assert!(!em_result.log_likelihood_history.is_empty());

        // Generate forecasts
        let forecasts = model.forecast(5).unwrap();
        assert_eq!(forecasts.len(), 5);
    }

    #[test]
    fn test_dynamic_linear_model_em() {
        // Create a simple AR(1) model in state-space form
        let transition = StateTransition {
            transition_matrix: array![[0.8]], // AR coefficient
            process_noise: array![[0.1]],
        };

        let observation = ObservationModel {
            observation_matrix: array![[1.0]],
            observation_noise: array![[0.1]],
        };

        let mut dlm = DynamicLinearModel::new(transition, observation);

        // Generate AR(1) test data
        let n = 50;
        let mut data = Array1::zeros(n);
        data[0] = 0.0;
        for i in 1..n {
            data[i] = 0.7 * data[i - 1] + 0.1 * (i as f64).sin();
        }

        // Fit using EM
        let em_result = dlm.fit(&data).unwrap();

        // Check convergence
        assert!(em_result.iterations > 0);
        assert!(!em_result.log_likelihood_history.is_empty());

        // Check that the estimated AR coefficient is reasonable
        let estimated_ar = dlm.transition.transition_matrix[[0, 0]];
        assert!(estimated_ar > 0.0 && estimated_ar < 1.0);
    }

    #[test]
    fn test_seasonal_model() {
        let model = StructuralModel::seasonal(0.1_f64, 0.05_f64, 0.2_f64, 4).unwrap();

        // Check structure
        assert!(model.level.is_some());
        assert!(model.seasonal.is_some());
        assert!(model.trend.is_none());

        // Check seasonal transition matrix dimensions
        let seasonal = model.seasonal.unwrap();
        assert_eq!(seasonal.transition_matrix.dim(), (3, 3)); // period - 1

        // First row should be all -1s
        for j in 0..3 {
            assert_relative_eq!(seasonal.transition_matrix[[0, j]], -1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_log_likelihood_computation() {
        let initial_state = StateVector {
            state: array![0.0],
            covariance: array![[1.0]],
        };

        let transition = StateTransition {
            transition_matrix: array![[0.9]],
            process_noise: array![[0.1]],
        };

        let observation = ObservationModel {
            observation_matrix: array![[1.0]],
            observation_noise: array![[0.1]],
        };

        let mut kf = KalmanFilter::new(initial_state, transition, observation);
        let observations = array![1.0, 0.9, 0.8, 0.7];

        let log_likelihood = kf.compute_log_likelihood(&observations).unwrap();

        // Log-likelihood should be finite and negative
        assert!(log_likelihood.is_finite());
        assert!(log_likelihood < 0.0);
    }
}
