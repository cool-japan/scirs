//! Recursive system identification for online applications
//!
//! This module provides recursive identification algorithms suitable for real-time
//! and online system identification applications where data arrives sequentially.

use crate::error::{SignalError, SignalResult};
use super::types::*;
use scirs2_core::ndarray::{Array1, Array2, Axis};

/// Recursive system identification for online applications
///
/// This structure implements recursive least squares (RLS) and related algorithms
/// for online system identification. It maintains parameter estimates and their
/// uncertainties as new data becomes available.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::Array1;
/// use scirs2_signal::sysid_enhanced::{RecursiveSysId, EnhancedSysIdConfig};
///
/// // Initialize with initial parameter guess
/// let initial_params = Array1::from_vec(vec![0.0, 0.0, 0.0]);
/// let config = EnhancedSysIdConfig::recursive();
/// let mut recursive_id = RecursiveSysId::new(initial_params, &config);
///
/// // Update with new data points
/// for (input, output) in input_data.iter().zip(output_data.iter()) {
///     let prediction_error = recursive_id.update(*input, *output)?;
///     println!("Prediction error: {:.4}", prediction_error);
/// }
///
/// // Get final parameter estimates
/// let params = recursive_id.get_parameters();
/// println!("Final parameters: {:?}", params);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct RecursiveSysId {
    /// Current parameter estimates
    parameters: Array1<f64>,
    /// Parameter covariance matrix (inverse of information matrix)
    covariance: Array2<f64>,
    /// Forgetting factor (0 < λ ≤ 1)
    lambda: f64,
    /// Buffer for regression vector construction
    phi_buffer: Vec<f64>,
    /// Model structure being identified
    structure: ModelStructure,
    /// Number of parameter updates performed
    n_updates: usize,
}

impl RecursiveSysId {
    /// Create a new recursive system identifier
    ///
    /// # Arguments
    ///
    /// * `initial_params` - Initial parameter estimates
    /// * `config` - Configuration containing forgetting factor and model structure
    ///
    /// # Returns
    ///
    /// * New RecursiveSysId instance ready for online identification
    pub fn new(initial_params: Array1<f64>, config: &EnhancedSysIdConfig) -> Self {
        let n_params = initial_params.len();

        Self {
            parameters: initial_params,
            covariance: Array2::eye(n_params) * 1000.0, // Large initial covariance
            lambda: config.forgetting_factor,
            phi_buffer: vec![0.0; n_params],
            structure: config.model_structure,
            n_updates: 0,
        }
    }

    /// Update parameter estimates with new input-output data point
    ///
    /// This implements the recursive least squares (RLS) algorithm:
    /// 1. Form regression vector from current and past data
    /// 2. Compute prediction using current parameter estimates
    /// 3. Update parameters based on prediction error
    /// 4. Update parameter covariance matrix
    ///
    /// # Arguments
    ///
    /// * `input` - New input sample
    /// * `output` - New output sample
    ///
    /// # Returns
    ///
    /// * Prediction error for the new sample
    ///
    /// # Errors
    ///
    /// Returns error if numerical computations fail or matrix becomes singular
    pub fn update(&mut self, input: f64, output: f64) -> SignalResult<f64> {
        // Update regression vector with new data
        self.update_regression_vector(input, output)?;

        let phi = Array1::from_vec(self.phi_buffer.clone());

        // Compute prediction with current parameters
        let y_pred = self.parameters.dot(&phi);
        let error = output - y_pred;

        // RLS parameter update
        let p_phi = self.covariance.dot(&phi);
        let denominator = self.lambda + phi.dot(&p_phi);

        if denominator.abs() > 1e-10 {
            // Compute Kalman gain
            let gain = &p_phi / denominator;

            // Update parameters: θ(k) = θ(k-1) + K(k) * e(k)
            self.parameters = &self.parameters + &gain * error;

            // Update covariance matrix: P(k) = [P(k-1) - K(k) * φᵀ(k) * P(k-1)] / λ
            let outer = gain
                .view()
                .insert_axis(Axis(1))
                .dot(&phi.view().insert_axis(Axis(0)));
            self.covariance = (&self.covariance - &outer.dot(&self.covariance)) / self.lambda;

            // Ensure covariance matrix remains symmetric and positive definite
            self.regularize_covariance();
        }

        self.n_updates += 1;

        Ok(error)
    }

    /// Update the regression vector based on model structure
    fn update_regression_vector(&mut self, input: f64, output: f64) -> SignalResult<()> {
        // Shift existing values in buffer (FIFO queue)
        for i in (1..self.phi_buffer.len()).rev() {
            self.phi_buffer[i] = self.phi_buffer[i - 1];
        }

        // Update based on model structure
        match self.structure {
            ModelStructure::ARX => {
                // ARX: φ(t) = [-y(t-1), ..., -y(t-na), u(t-1), ..., u(t-nb)]
                let na = self.phi_buffer.len() / 2;
                if self.phi_buffer.len() > 0 {
                    self.phi_buffer[0] = -output;
                }
                if self.phi_buffer.len() > na {
                    self.phi_buffer[na] = input;
                }
            }
            ModelStructure::ARMAX => {
                // ARMAX: φ(t) = [-y(t-1), ..., -y(t-na), u(t-1), ..., u(t-nb), e(t-1), ..., e(t-nc)]
                // For simplicity, using output error as noise approximation
                self.phi_buffer[0] = -output;
                if self.phi_buffer.len() > 1 {
                    self.phi_buffer[1] = input;
                }
                // Note: proper ARMAX requires maintaining error history
            }
            _ => {
                // Default: simple ARX-like structure
                self.phi_buffer[0] = -output;
                if self.phi_buffer.len() > 1 {
                    self.phi_buffer[1] = input;
                }
            }
        }

        Ok(())
    }

    /// Regularize covariance matrix to maintain numerical stability
    fn regularize_covariance(&mut self) {
        // Add small diagonal regularization to prevent numerical issues
        let n = self.covariance.nrows();
        let regularization = Array2::eye(n) * 1e-12;
        self.covariance = &self.covariance + &regularization;

        // Ensure symmetry (important for numerical stability)
        self.covariance = (&self.covariance + &self.covariance.t()) * 0.5;
    }

    /// Get current parameter estimates
    ///
    /// # Returns
    ///
    /// * Reference to current parameter vector
    pub fn get_parameters(&self) -> &Array1<f64> {
        &self.parameters
    }

    /// Get parameter standard errors (uncertainties)
    ///
    /// # Returns
    ///
    /// * Array of parameter standard errors (square root of diagonal covariance elements)
    pub fn get_uncertainties(&self) -> Array1<f64> {
        self.covariance.diag().map(|x| x.sqrt().max(0.0))
    }

    /// Get parameter covariance matrix
    ///
    /// # Returns
    ///
    /// * Reference to parameter covariance matrix
    pub fn get_covariance(&self) -> &Array2<f64> {
        &self.covariance
    }

    /// Get number of updates performed
    ///
    /// # Returns
    ///
    /// * Number of parameter updates
    pub fn get_update_count(&self) -> usize {
        self.n_updates
    }

    /// Reset the identifier to initial conditions
    ///
    /// # Arguments
    ///
    /// * `initial_params` - New initial parameter values
    /// * `initial_covariance` - Optional initial covariance (defaults to large diagonal)
    pub fn reset(&mut self, initial_params: Array1<f64>, initial_covariance: Option<Array2<f64>>) {
        let n_params = initial_params.len();

        self.parameters = initial_params;
        self.covariance = initial_covariance.unwrap_or_else(|| Array2::eye(n_params) * 1000.0);
        self.phi_buffer = vec![0.0; n_params];
        self.n_updates = 0;
    }

    /// Set forgetting factor
    ///
    /// # Arguments
    ///
    /// * `lambda` - New forgetting factor (0 < λ ≤ 1)
    ///
    /// # Errors
    ///
    /// Returns error if lambda is not in valid range
    pub fn set_forgetting_factor(&mut self, lambda: f64) -> SignalResult<()> {
        if lambda <= 0.0 || lambda > 1.0 {
            return Err(SignalError::ValueError(format!(
                "Forgetting factor must be in (0, 1], got {}",
                lambda
            )));
        }
        self.lambda = lambda;
        Ok(())
    }

    /// Get current forgetting factor
    pub fn get_forgetting_factor(&self) -> f64 {
        self.lambda
    }

    /// Compute current parameter confidence intervals
    ///
    /// # Arguments
    ///
    /// * `confidence_level` - Confidence level (e.g., 0.95 for 95% intervals)
    ///
    /// # Returns
    ///
    /// * Vector of (lower, upper) confidence interval pairs
    pub fn get_confidence_intervals(&self, confidence_level: f64) -> Vec<(f64, f64)> {
        let alpha = 1.0 - confidence_level;
        let z_score = normal_inverse_cdf(1.0 - alpha / 2.0); // Approximate normal quantile

        let std_errors = self.get_uncertainties();

        self.parameters
            .iter()
            .zip(std_errors.iter())
            .map(|(&param, &std_err)| {
                let margin = z_score * std_err;
                (param - margin, param + margin)
            })
            .collect()
    }

    /// Predict output for given input using current model
    ///
    /// # Arguments
    ///
    /// * `input` - Input value for prediction
    ///
    /// # Returns
    ///
    /// * Predicted output value
    pub fn predict(&self, input: f64) -> f64 {
        // Create regression vector for prediction
        let mut phi = self.phi_buffer.clone();

        // Update with current input (structure-dependent)
        match self.structure {
            ModelStructure::ARX => {
                let na = phi.len() / 2;
                if phi.len() > na {
                    phi[na] = input;
                }
            }
            _ => {
                if phi.len() > 1 {
                    phi[1] = input;
                }
            }
        }

        let phi_array = Array1::from_vec(phi);
        self.parameters.dot(&phi_array)
    }

    /// Compute information matrix condition number
    ///
    /// # Returns
    ///
    /// * Condition number of current information matrix (inverse of covariance)
    pub fn compute_condition_number(&self) -> f64 {
        // Condition number is ratio of largest to smallest eigenvalue
        // For simplicity, use trace/determinant approximation
        let trace = self.covariance.diag().sum();
        let det_approx = self.covariance.diag().iter().product::<f64>().abs();

        if det_approx > 1e-15 {
            trace / det_approx.powf(1.0 / self.covariance.nrows() as f64)
        } else {
            f64::INFINITY
        }
    }
}

/// Approximate normal inverse CDF for confidence intervals
fn normal_inverse_cdf(p: f64) -> f64 {
    // Beasley-Springer-Moro approximation for normal quantile
    let a0 = -3.969683028665376e+01;
    let a1 = 2.209460984245205e+02;
    let a2 = -2.759285104469687e+02;
    let a3 = 1.383577518672690e+02;
    let a4 = -3.066479806614716e+01;
    let a5 = 2.506628277459239e+00;

    let b1 = -5.447609879822406e+01;
    let b2 = 1.615858368580409e+02;
    let b3 = -1.556989798598866e+02;
    let b4 = 6.680131188771972e+01;
    let b5 = -1.328068155288572e+01;

    let c0 = -7.784894002430293e-03;
    let c1 = -3.223964580411365e-01;
    let c2 = -2.400758277161838e+00;
    let c3 = -2.549732539343734e+00;
    let c4 = 4.374664141464968e+00;
    let c5 = 2.938163982698783e+00;

    let d1 = 7.784695709041462e-03;
    let d2 = 3.224671290700398e-01;
    let d3 = 2.445134137142996e+00;
    let d4 = 3.754408661907416e+00;

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    let q = if p < p_low {
        p
    } else if p > p_high {
        1.0 - p
    } else {
        0.5 - (p - 0.5).abs()
    };

    let (x, sign) = if p < p_low {
        let r = (-2.0 * q.ln()).sqrt();
        let x = (((((c5 * r + c4) * r + c3) * r + c2) * r + c1) * r + c0)
            / ((((d4 * r + d3) * r + d2) * r + d1) * r + 1.0);
        (x, -1.0)
    } else if p > p_high {
        let r = (-2.0 * q.ln()).sqrt();
        let x = (((((c5 * r + c4) * r + c3) * r + c2) * r + c1) * r + c0)
            / ((((d4 * r + d3) * r + d2) * r + d1) * r + 1.0);
        (x, 1.0)
    } else {
        let r = (p - 0.5) * (p - 0.5);
        let x = (p - 0.5) * (((((a5 * r + a4) * r + a3) * r + a2) * r + a1) * r + a0)
            / (((((b5 * r + b4) * r + b3) * r + b2) * r + b1) * r + 1.0);
        (x, 1.0)
    };

    sign * x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recursive_creation() {
        let initial_params = Array1::from_vec(vec![0.5, -0.2, 0.8]);
        let config = EnhancedSysIdConfig::recursive();
        let recursive_id = RecursiveSysId::new(initial_params.clone(), &config);

        assert_eq!(recursive_id.get_parameters(), &initial_params);
        assert_eq!(recursive_id.get_update_count(), 0);
    }

    #[test]
    fn test_recursive_update() {
        let initial_params = Array1::from_vec(vec![0.0, 0.0]);
        let config = EnhancedSysIdConfig::recursive();
        let mut recursive_id = RecursiveSysId::new(initial_params, &config);

        // Test single update
        let error = recursive_id.update(1.0, 0.5);
        assert!(error.is_ok());
        assert_eq!(recursive_id.get_update_count(), 1);

        // Parameters should have changed
        let updated_params = recursive_id.get_parameters();
        assert!(updated_params.iter().any(|&x| x.abs() > 1e-10));
    }

    #[test]
    fn test_forgetting_factor_validation() {
        let initial_params = Array1::from_vec(vec![0.0, 0.0]);
        let config = EnhancedSysIdConfig::recursive();
        let mut recursive_id = RecursiveSysId::new(initial_params, &config);

        // Valid forgetting factor
        assert!(recursive_id.set_forgetting_factor(0.95).is_ok());

        // Invalid forgetting factors
        assert!(recursive_id.set_forgetting_factor(0.0).is_err());
        assert!(recursive_id.set_forgetting_factor(1.1).is_err());
    }

    #[test]
    fn test_confidence_intervals() {
        let initial_params = Array1::from_vec(vec![1.0, -0.5]);
        let config = EnhancedSysIdConfig::recursive();
        let recursive_id = RecursiveSysId::new(initial_params, &config);

        let intervals = recursive_id.get_confidence_intervals(0.95);
        assert_eq!(intervals.len(), 2);

        // Each interval should be centered around the parameter value
        for (i, &param) in recursive_id.get_parameters().iter().enumerate() {
            let (lower, upper) = intervals[i];
            assert!(lower < param);
            assert!(param < upper);
        }
    }

    #[test]
    fn test_prediction() {
        let initial_params = Array1::from_vec(vec![0.5, 0.3]);
        let config = EnhancedSysIdConfig::recursive();
        let recursive_id = RecursiveSysId::new(initial_params, &config);

        let prediction = recursive_id.predict(1.0);
        assert!(prediction.is_finite());
    }
}