// Recursive identification methods for system identification

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array1, Array2, Axis};

/// Recursive least squares implementation
#[derive(Debug, Clone)]
pub struct RecursiveLeastSquares {
    /// Current parameter estimates
    pub parameters: Array1<f64>,
    /// Covariance matrix
    pub covariance: Array2<f64>,
    /// Forgetting factor
    pub forgetting_factor: f64,
    /// Parameter dimension
    pub dimension: usize,
}

impl RecursiveLeastSquares {
    /// Create new RLS estimator
    ///
    /// # Arguments
    /// * `dimension` - Number of parameters to estimate
    /// * `forgetting_factor` - Forgetting factor (0 < λ ≤ 1)
    /// * `initial_covariance` - Initial covariance scaling
    ///
    /// # Returns
    /// * New RLS estimator
    pub fn new(dimension: usize, forgetting_factor: f64, initial_covariance: f64) -> Self {
        let parameters = Array1::<f64>::zeros(dimension);
        let covariance = Array2::<f64>::eye(dimension) * initial_covariance;

        Self {
            parameters,
            covariance,
            forgetting_factor,
            dimension,
        }
    }

    /// Update estimates with new data point
    ///
    /// # Arguments
    /// * `regression_vector` - Input regression vector
    /// * `output` - Corresponding output value
    ///
    /// # Returns
    /// * Prediction error
    pub fn update(&mut self, regression_vector: &Array1<f64>, output: f64) -> SignalResult<f64> {
        if regression_vector.len() != self.dimension {
            return Err(SignalError::ValueError(
                "Regression _vector dimension mismatch".to_string(),
            ));
        }

        // Prediction error
        let prediction = self.parameters.dot(regression_vector);
        let error = output - prediction;

        // Gain _vector: K = P * phi / (lambda + phi^T * P * phi)
        let p_phi = self.covariance.dot(regression_vector);
        let denominator = self.forgetting_factor + regression_vector.dot(&p_phi);

        if denominator.abs() < 1e-12 {
            return Err(SignalError::ComputationError(
                "RLS update encountered numerical issues".to_string(),
            ));
        }

        let gain = &p_phi / denominator;

        // Parameter update: θ = θ + K * error
        let parameter_update = &gain * error;
        self.parameters += &parameter_update;

        // Covariance update: P = (P - K * phi^T * P) / lambda
        let k_phi_t_p = gain.insert_axis(Axis(1)).dot(
            &regression_vector
                .clone()
                .insert_axis(Axis(0))
                .dot(&self.covariance),
        );
        self.covariance = (&self.covariance - &k_phi_t_p) / self.forgetting_factor;

        Ok(error)
    }

    /// Get current parameter estimates
    pub fn get_parameters(&self) -> &Array1<f64> {
        &self.parameters
    }

    /// Get parameter covariance matrix
    pub fn get_covariance(&self) -> &Array2<f64> {
        &self.covariance
    }
}
