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

/// Simple pseudo-inverse implementation using SVD
fn pseudo_inverse<F>(matrix: &Array2<F>) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Display + ScalarOperand,
{
    // This is a placeholder implementation
    // In practice, we would use a proper SVD-based pseudo-inverse
    let (m, n) = matrix.dim();
    if m != n {
        return Err(TimeSeriesError::InvalidInput(
            "Matrix must be square for pseudo-inverse".to_string(),
        ));
    }

    // Simple check for diagonal dominance
    let mut is_diagonal = true;
    for i in 0..m {
        for j in 0..n {
            if i != j && matrix[[i, j]].abs() > F::epsilon() {
                is_diagonal = false;
                break;
            }
        }
    }

    if is_diagonal {
        // For diagonal matrices, invert the diagonal elements
        let mut inv = Array2::zeros((m, n));
        for i in 0..m {
            let diag = matrix[[i, i]];
            if diag.abs() > F::epsilon() {
                inv[[i, i]] = F::one() / diag;
            }
        }
        Ok(inv)
    } else {
        // Placeholder for general case - would need proper linear algebra
        Err(TimeSeriesError::ComputationError(
            "General matrix inversion not yet implemented".to_string(),
        ))
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

    /// Fit the model to data using maximum likelihood
    pub fn fit<S>(&mut self, data: &ArrayBase<S, Ix1>) -> Result<()>
    where
        S: Data<Elem = F>,
    {
        scirs2_core::validation::check_array_finite(data, "data")?;

        // Placeholder implementation
        // In practice, we would use MLE or EM algorithm
        Ok(())
    }

    /// Forecast using the fitted model
    pub fn forecast(&self, steps: usize) -> Result<Array1<F>> {
        // Placeholder implementation
        Ok(Array1::zeros(steps))
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
    pub fn fit<S>(&mut self, data: &ArrayBase<S, Ix1>) -> Result<()>
    where
        S: Data<Elem = F>,
    {
        scirs2_core::validation::check_array_finite(data, "data")?;

        // Placeholder for EM algorithm implementation
        Ok(())
    }

    /// Smooth the state estimates
    pub fn smooth(&self, _observations: &Array1<F>) -> Result<Vec<StateVector<F>>> {
        // Placeholder for Kalman smoothing
        Ok(vec![])
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
}
