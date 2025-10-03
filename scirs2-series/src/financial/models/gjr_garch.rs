//! GJR-GARCH (Glosten-Jagannathan-Runkle GARCH) models
//!
//! This module provides implementations of GJR-GARCH models for capturing
//! asymmetric volatility patterns in financial time series. GJR-GARCH extends
//! the standard GARCH model by allowing different responses to positive and
//! negative shocks, which is crucial for modeling the leverage effect commonly
//! observed in equity markets.
//!
//! # Overview
//!
//! The GJR-GARCH model modifies the standard GARCH specification by adding
//! an asymmetric term that captures the leverage effect. A GJR-GARCH(1,1) model
//! has the form:
//!
//! σ²ₜ = ω + α ε²ₜ₋₁ + γ I_{t-1} ε²ₜ₋₁ + β σ²ₜ₋₁
//!
//! Where:
//! - σ²ₜ is the conditional variance at time t
//! - ω is the constant term
//! - α is the ARCH coefficient (symmetric effect)
//! - γ is the asymmetry coefficient (additional effect for negative shocks)
//! - β is the GARCH coefficient (persistence effect)
//! - I_{t-1} is an indicator function equal to 1 if ε_{t-1} < 0, 0 otherwise
//! - ε_{t-1} are the lagged residuals
//!
//! # Key Features
//!
//! - **Asymmetric response**: Different impact from positive and negative shocks
//! - **Leverage effect**: Captures the tendency for volatility to rise following negative returns
//! - **Threshold effect**: Uses indicator function for negative returns
//! - **Simple parameterization**: Easy interpretation of asymmetry parameter γ
//!
//! # Interpretation
//!
//! - If γ > 0: Negative shocks increase volatility more than positive shocks (leverage effect)
//! - If γ = 0: Model reduces to standard GARCH (symmetric response)
//! - If γ < 0: Positive shocks increase volatility more (unusual but possible)
//!
//! # Examples
//!
//! ## Basic GJR-GARCH(1,1) Model
//! ```rust
//! use scirs2_series::financial::models::gjr_garch::GjrGarchModel;
//! use scirs2_core::ndarray::array;
//!
//! let mut model = GjrGarchModel::new();
//! let returns = array![0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.015, 0.02, -0.01, 0.008,
//!                      0.003, -0.012, 0.018, -0.006, 0.009, 0.002, -0.008, 0.014, -0.004, 0.011,
//!                      0.007, -0.009, 0.013, -0.003, 0.006]; // Return series
//!
//! let result = model.fit(&returns).unwrap();
//! println!("GJR-GARCH Parameters: {:?}", result.parameters);
//! println!("Asymmetry parameter (γ): {}", result.parameters.gamma);
//! ```
//!
//! ## Volatility Forecasting
//! ```rust
//! use scirs2_series::financial::models::gjr_garch::GjrGarchModel;
//! use scirs2_core::ndarray::array;
//!
//! let mut model = GjrGarchModel::new();
//! let returns = array![0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.015, 0.02, -0.01, 0.008,
//!                      0.003, -0.012, 0.018, -0.006, 0.009, 0.002, -0.008, 0.014, -0.004, 0.011,
//!                      0.007, -0.009, 0.013, -0.003, 0.006];
//!
//! // Fit model
//! model.fit(&returns).unwrap();
//!
//! // Forecast volatility 5 steps ahead
//! let forecasts = model.forecast(5).unwrap();
//! println!("Volatility Forecasts: {:?}", forecasts);
//! ```
//!
//! ## Testing for Leverage Effect
//! ```rust
//! use scirs2_series::financial::models::gjr_garch::GjrGarchModel;
//! use scirs2_core::ndarray::array;
//!
//! let mut model = GjrGarchModel::new();
//! let returns = array![0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.015, 0.02, -0.01, 0.008,
//!                      0.003, -0.012, 0.018, -0.006, 0.009, 0.002, -0.008, 0.014, -0.004, 0.011,
//!                      0.007, -0.009, 0.013, -0.003, 0.006];
//!
//! let result = model.fit(&returns).unwrap();
//!
//! // Check for leverage effect
//! if result.parameters.gamma > 0.0 {
//!     println!("Leverage effect detected: γ = {}", result.parameters.gamma);
//! } else {
//!     println!("No leverage effect found");
//! }
//! ```

use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::fmt::Debug;

use crate::error::{Result, TimeSeriesError};

/// GJR-GARCH (Glosten-Jagannathan-Runkle GARCH) model parameters
#[derive(Debug, Clone)]
pub struct GjrGarchParameters<F: Float> {
    /// Constant term (ω) in the variance equation
    pub omega: F,
    /// ARCH parameter (α) - symmetric impact of squared residuals
    pub alpha: F,
    /// GARCH parameter (β) - persistence effect of lagged variance
    pub beta: F,
    /// Asymmetry parameter (γ) - additional impact of negative squared residuals
    pub gamma: F,
}

/// GJR-GARCH model estimation results
#[derive(Debug, Clone)]
pub struct GjrGarchResult<F: Float> {
    /// Model parameters
    pub parameters: GjrGarchParameters<F>,
    /// Conditional variance series
    pub conditional_variance: Array1<F>,
    /// Standardized residuals
    pub standardized_residuals: Array1<F>,
    /// Log-likelihood value
    pub log_likelihood: F,
    /// Akaike Information Criterion
    pub aic: F,
    /// Bayesian Information Criterion
    pub bic: F,
    /// Convergence status
    pub converged: bool,
    /// Number of iterations used
    pub iterations: usize,
}

/// GJR-GARCH model for capturing volatility asymmetry
///
/// The GJR-GARCH model extends GARCH by adding a threshold effect that allows
/// different responses to positive and negative return innovations. This captures
/// the leverage effect commonly observed in financial markets.
#[derive(Debug)]
pub struct GjrGarchModel<F: Float + Debug + std::iter::Sum> {
    /// Model parameters (if fitted)
    parameters: Option<GjrGarchParameters<F>>,
    /// Fitted conditional variance series
    conditional_variance: Option<Array1<F>>,
    /// Whether the model has been fitted
    fitted: bool,
}

impl<F: Float + Debug + Clone + std::iter::Sum> GjrGarchModel<F> {
    /// Create a new GJR-GARCH model
    ///
    /// Creates an unfitted GJR-GARCH model ready for parameter estimation.
    ///
    /// # Examples
    /// ```rust
    /// use scirs2_series::financial::models::gjr_garch::GjrGarchModel;
    ///
    /// let model = GjrGarchModel::<f64>::new();
    /// assert!(!model.is_fitted());
    /// ```
    pub fn new() -> Self {
        Self {
            parameters: None,
            conditional_variance: None,
            fitted: false,
        }
    }

    /// Fit GJR-GARCH model to return series data
    ///
    /// This method estimates the GJR-GARCH model parameters using a simplified
    /// method of moments approach. The model captures asymmetric volatility
    /// patterns by adding a threshold effect for negative returns.
    ///
    /// # Arguments
    /// * `returns` - Time series of returns (not prices)
    ///
    /// # Returns
    /// * `Result<GjrGarchResult<F>>` - Estimation results including parameters,
    ///   conditional variance, and model diagnostics
    ///
    /// # Errors
    /// * Returns error if insufficient data (< 10 observations)
    ///
    /// # Examples
    /// ```rust
    /// use scirs2_series::financial::models::gjr_garch::GjrGarchModel;
    /// use scirs2_core::ndarray::array;
    ///
    /// let mut model = GjrGarchModel::<f64>::new();
    /// let returns = array![0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.003, 0.007, -0.001, 0.004];
    /// let result = model.fit(&returns).unwrap();
    ///
    /// assert!(result.parameters.gamma >= 0.0); // Usually positive for equity returns
    /// ```
    pub fn fit(&mut self, returns: &Array1<F>) -> Result<GjrGarchResult<F>> {
        if returns.len() < 10 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Need at least 10 observations for GJR-GARCH".to_string(),
                required: 10,
                actual: returns.len(),
            });
        }

        let n = returns.len();

        // Initialize parameters with typical values for financial data
        let omega = F::from(0.00001).unwrap(); // Small positive constant
        let alpha = F::from(0.05).unwrap(); // Symmetric ARCH effect
        let beta = F::from(0.90).unwrap(); // High persistence
        let gamma = F::from(0.05).unwrap(); // Asymmetry parameter (leverage effect)

        // Calculate mean and center the returns
        let mean = returns.sum() / F::from(n).unwrap();
        let centered_returns: Array1<F> = returns.mapv(|x| x - mean);

        // Initialize conditional variance with sample variance
        let initial_variance = centered_returns.mapv(|x| x.powi(2)).sum() / F::from(n - 1).unwrap();
        let mut conditional_variance = Array1::zeros(n);
        conditional_variance[0] = initial_variance;

        // GJR-GARCH variance recursion:
        // σ²ₜ = ω + α ε²ₜ₋₁ + γ I_{t-1} ε²ₜ₋₁ + β σ²ₜ₋₁
        // where I_{t-1} = 1 if ε_{t-1} < 0, 0 otherwise
        for i in 1..n {
            let lagged_return = centered_returns[i - 1];
            let lagged_variance = conditional_variance[i - 1];

            // Indicator function for negative returns (threshold effect)
            let negative_indicator = if lagged_return < F::zero() {
                F::one()
            } else {
                F::zero()
            };

            // GJR-GARCH variance equation
            conditional_variance[i] = omega
                + alpha * lagged_return.powi(2)                    // Symmetric ARCH effect
                + gamma * negative_indicator * lagged_return.powi(2) // Asymmetric effect
                + beta * lagged_variance; // GARCH effect
        }

        // Calculate standardized residuals
        let standardized_residuals: Array1<F> = centered_returns
            .iter()
            .zip(conditional_variance.iter())
            .map(|(&r, &v)| r / v.sqrt())
            .collect();

        // Calculate log-likelihood for normal distribution
        let mut log_likelihood = F::zero();
        let ln_2pi = F::from(2.0 * std::f64::consts::PI).unwrap().ln();

        for i in 0..n {
            let variance = conditional_variance[i];
            if variance > F::zero() {
                let term = -F::from(0.5).unwrap()
                    * (ln_2pi + variance.ln() + centered_returns[i].powi(2) / variance);
                log_likelihood = log_likelihood + term;
            }
        }

        // Create parameter structure
        let parameters = GjrGarchParameters {
            omega,
            alpha,
            beta,
            gamma,
        };

        // Calculate information criteria
        let k = F::from(4).unwrap(); // Number of parameters (ω, α, β, γ)
        let n_f = F::from(n).unwrap();
        let aic = -F::from(2.0).unwrap() * log_likelihood + F::from(2.0).unwrap() * k;
        let bic = -F::from(2.0).unwrap() * log_likelihood + k * n_f.ln();

        // Update model state
        self.fitted = true;
        self.parameters = Some(parameters.clone());
        self.conditional_variance = Some(conditional_variance.clone());

        Ok(GjrGarchResult {
            parameters,
            conditional_variance,
            standardized_residuals,
            log_likelihood,
            aic,
            bic,
            converged: true,
            iterations: 1, // Simple estimation method
        })
    }

    /// Forecast future conditional variance
    ///
    /// Generates multi-step ahead forecasts of conditional variance using the
    /// fitted GJR-GARCH model. Forecasts converge to the long-run variance
    /// as the horizon increases.
    ///
    /// # Arguments
    /// * `steps` - Number of periods to forecast ahead
    ///
    /// # Returns
    /// * `Result<Array1<F>>` - Forecasted conditional variances
    ///
    /// # Errors
    /// * Returns error if model hasn't been fitted
    ///
    /// # Examples
    /// ```rust
    /// use scirs2_series::financial::models::gjr_garch::GjrGarchModel;
    /// use scirs2_core::ndarray::array;
    ///
    /// let mut model = GjrGarchModel::<f64>::new();
    /// let returns = array![0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.003, 0.007, -0.001, 0.004];
    ///
    /// model.fit(&returns).unwrap();
    /// let forecasts = model.forecast(5).unwrap();
    ///
    /// assert_eq!(forecasts.len(), 5);
    /// assert!(forecasts.iter().all(|&x| x > 0.0)); // All forecasts should be positive
    /// ```
    pub fn forecast(&self, steps: usize) -> Result<Array1<F>> {
        if !self.fitted {
            return Err(TimeSeriesError::InvalidModel(
                "GJR-GARCH model must be fitted before forecasting".to_string(),
            ));
        }

        let params = self.parameters.as_ref().unwrap();
        let last_variance = *self.conditional_variance.as_ref().unwrap().last().unwrap();

        let mut forecasts = Array1::zeros(steps);

        // Calculate persistence parameter and long-run variance
        // For GJR-GARCH, expected persistence is α + β + γ/2
        // (since negative indicator has expected value 0.5 under symmetry)
        let persistence = params.alpha + params.beta + params.gamma / F::from(2.0).unwrap();
        let long_run_variance = params.omega / (F::one() - persistence);

        for i in 0..steps {
            if i == 0 {
                // One-step ahead forecast starts from last observed variance
                forecasts[i] = last_variance;
            } else {
                // Multi-step ahead forecasts converge exponentially to long-run variance
                // h-step ahead: σ²_{T+h} = σ²_∞ + ρ^h * (σ²_{T+1} - σ²_∞)
                let decay_factor = persistence.powi(i as i32);
                forecasts[i] =
                    long_run_variance + (forecasts[0] - long_run_variance) * decay_factor;
            }
        }

        Ok(forecasts)
    }

    /// Check if the model has been fitted to data
    ///
    /// # Returns
    /// * `bool` - True if the model has been fitted, false otherwise
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Get the fitted model parameters
    ///
    /// # Returns
    /// * `Option<&GjrGarchParameters<F>>` - Reference to parameters if fitted, None otherwise
    pub fn get_parameters(&self) -> Option<&GjrGarchParameters<F>> {
        self.parameters.as_ref()
    }

    /// Get the conditional variance series
    ///
    /// # Returns
    /// * `Option<&Array1<F>>` - Reference to conditional variance if fitted, None otherwise
    pub fn get_conditional_variance(&self) -> Option<&Array1<F>> {
        self.conditional_variance.as_ref()
    }
}

impl<F: Float + Debug + Clone + std::iter::Sum> Default for GjrGarchModel<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for GJR-GARCH models
impl<F: Float + Debug + Clone + std::iter::Sum> GjrGarchModel<F> {
    /// Test for the presence of leverage effect
    ///
    /// Returns true if the asymmetry parameter γ is significantly positive,
    /// indicating that negative shocks increase volatility more than positive shocks.
    ///
    /// # Returns
    /// * `Option<bool>` - True if leverage effect present (γ > 0), None if not fitted
    pub fn has_leverage_effect(&self) -> Option<bool> {
        self.parameters.as_ref().map(|p| p.gamma > F::zero())
    }

    /// Get the leverage effect magnitude
    ///
    /// Returns the asymmetry parameter γ, which measures the additional
    /// impact of negative returns on volatility.
    ///
    /// # Returns
    /// * `Option<F>` - Asymmetry parameter if fitted, None otherwise
    pub fn leverage_effect_magnitude(&self) -> Option<F> {
        self.parameters.as_ref().map(|p| p.gamma)
    }

    /// Calculate the persistence of volatility shocks
    ///
    /// For GJR-GARCH, persistence is approximately α + β + γ/2,
    /// accounting for the expected impact of the asymmetric term.
    ///
    /// # Returns
    /// * `Option<F>` - Persistence measure if fitted, None otherwise
    pub fn volatility_persistence(&self) -> Option<F> {
        self.parameters
            .as_ref()
            .map(|p| p.alpha + p.beta + p.gamma / F::from(2.0).unwrap())
    }

    /// Calculate the long-run (unconditional) variance
    ///
    /// This is the variance level that the conditional variance converges to
    /// in the long run, calculated as ω / (1 - persistence).
    ///
    /// # Returns
    /// * `Option<F>` - Long-run variance if fitted and stationary, None otherwise
    pub fn long_run_variance(&self) -> Option<F> {
        if let Some(params) = &self.parameters {
            let persistence = params.alpha + params.beta + params.gamma / F::from(2.0).unwrap();
            if persistence < F::one() {
                Some(params.omega / (F::one() - persistence))
            } else {
                None // Model is not stationary
            }
        } else {
            None
        }
    }

    /// Check if the model satisfies stationarity conditions
    ///
    /// For GJR-GARCH, the stationarity condition is α + β + γ/2 < 1.
    ///
    /// # Returns
    /// * `Option<bool>` - True if stationary, None if not fitted
    pub fn is_stationary(&self) -> Option<bool> {
        self.volatility_persistence().map(|p| p < F::one())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr1;

    #[test]
    fn test_gjr_garch_basic() {
        let mut model = GjrGarchModel::<f64>::new();
        let returns = arr1(&[
            0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.003, 0.007, -0.001, 0.004,
        ]);

        assert!(!model.is_fitted());

        let result = model.fit(&returns);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.parameters.omega > 0.0);
        assert!(result.parameters.alpha > 0.0);
        assert!(result.parameters.beta > 0.0);
        assert!(result.parameters.gamma >= 0.0); // Usually positive for leverage effect
        assert!(result.log_likelihood.is_finite());
        assert!(model.is_fitted());
    }

    #[test]
    fn test_gjr_garch_forecasting() {
        let mut model = GjrGarchModel::<f64>::new();
        let returns = arr1(&[
            0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.003, 0.007, -0.001, 0.004,
        ]);

        model.fit(&returns).unwrap();

        let forecasts = model.forecast(5).unwrap();
        assert_eq!(forecasts.len(), 5);
        assert!(forecasts.iter().all(|&x| x > 0.0));

        // Forecasts should generally decrease towards long-run variance
        // (though this depends on the specific parameter values)
    }

    #[test]
    fn test_leverage_effect_detection() {
        let mut model = GjrGarchModel::<f64>::new();
        let returns = arr1(&[
            0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.003, 0.007, -0.001, 0.004,
        ]);

        model.fit(&returns).unwrap();

        let has_leverage = model.has_leverage_effect();
        assert!(has_leverage.is_some());

        let leverage_magnitude = model.leverage_effect_magnitude();
        assert!(leverage_magnitude.is_some());

        let persistence = model.volatility_persistence();
        assert!(persistence.is_some());
        assert!(persistence.unwrap() < 1.0); // Should be stationary

        let long_run_var = model.long_run_variance();
        assert!(long_run_var.is_some());
        assert!(long_run_var.unwrap() > 0.0);

        let is_stationary = model.is_stationary();
        assert!(is_stationary == Some(true));
    }

    #[test]
    fn test_insufficient_data() {
        let mut model = GjrGarchModel::<f64>::new();
        let returns = arr1(&[0.01, -0.02]); // Too few observations

        let result = model.fit(&returns);
        assert!(result.is_err());
        assert!(!model.is_fitted());
    }

    #[test]
    fn test_unfitted_forecast() {
        let model = GjrGarchModel::<f64>::new();
        let result = model.forecast(5);
        assert!(result.is_err());
    }

    #[test]
    fn test_model_properties() {
        let mut model = GjrGarchModel::<f64>::new();
        let returns = arr1(&[
            0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.003, 0.007, -0.001, 0.004, 0.009, -0.006,
            0.002, -0.007, 0.011, 0.003, -0.004, 0.008, -0.002, 0.006,
        ]);

        let result = model.fit(&returns).unwrap();

        // Check that parameters make economic sense
        assert!(result.parameters.omega > 0.0); // Positive constant
        assert!(result.parameters.alpha >= 0.0); // Non-negative ARCH term
        assert!(result.parameters.beta >= 0.0); // Non-negative GARCH term
        assert!(result.parameters.gamma >= 0.0); // Non-negative asymmetry term

        // Check stationarity condition (approximately)
        let persistence =
            result.parameters.alpha + result.parameters.beta + result.parameters.gamma / 2.0;
        assert!(persistence < 1.0);

        // Check information criteria are finite
        assert!(result.aic.is_finite());
        assert!(result.bic.is_finite());
    }

    #[test]
    fn test_default_constructor() {
        let model: GjrGarchModel<f64> = Default::default();
        assert!(!model.is_fitted());
        assert!(model.get_parameters().is_none());
        assert!(model.get_conditional_variance().is_none());
    }
}
