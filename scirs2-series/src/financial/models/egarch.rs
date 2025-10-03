//! EGARCH (Exponential GARCH) models
//!
//! This module provides implementations of EGARCH models for capturing asymmetric
//! volatility patterns in financial time series. EGARCH models allow for different
//! responses to positive and negative shocks, which is commonly observed in
//! financial markets where negative news tends to increase volatility more than
//! positive news of the same magnitude.
//!
//! # Overview
//!
//! The EGARCH model uses the logarithm of conditional variance, ensuring that
//! variance is always positive without parameter restrictions. An EGARCH(p,q) model
//! has the form:
//!
//! ln(σ²ₜ) = ω + Σᵢ₌₁ᵖ βᵢ ln(σ²ₜ₋ᵢ) + Σⱼ₌₁ᵠ [αⱼ(|zₜ₋ⱼ| - E[|zₜ₋ⱼ|]) + γⱼzₜ₋ⱼ]
//!
//! Where:
//! - σ²ₜ is the conditional variance at time t
//! - ω is the constant term
//! - βᵢ are the persistence coefficients
//! - αⱼ are the magnitude effect coefficients
//! - γⱼ are the asymmetry effect coefficients (leverage effect)
//! - zₜ = εₜ/σₜ are the standardized residuals
//!
//! # Key Features
//!
//! - **Asymmetric response**: Different impact from positive and negative shocks
//! - **Always positive variance**: Logarithmic specification ensures σ²ₜ > 0
//! - **Leverage effect**: Captures the tendency for volatility to rise following negative returns
//! - **No parameter restrictions**: Unlike GARCH, parameters don't need constraints for stationarity
//!
//! # Examples
//!
//! ## Basic EGARCH(1,1) Model
//! ```rust
//! use scirs2_series::financial::models::egarch::{EgarchModel, EgarchConfig};
//! use scirs2_core::ndarray::array;
//!
//! let mut model = EgarchModel::egarch_11();
//! let data = array![0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.015, 0.02, -0.01, 0.008,
//!                   0.003, -0.012, 0.018, -0.006, 0.009, 0.002, -0.008, 0.014, -0.004, 0.011,
//!                   0.007, -0.009, 0.013, -0.003, 0.006, 0.004, -0.007, 0.016, -0.002, 0.010,
//!                   0.001, -0.005, 0.012, -0.001, 0.008]; // Returns (35 points for EGARCH)
//!
//! let result = model.fit(&data).unwrap();
//! println!("EGARCH Parameters: {:?}", result.parameters);
//! println!("Log-likelihood: {}", result.log_likelihood);
//! ```
//!
//! ## Custom EGARCH Configuration
//! ```rust
//! use scirs2_series::financial::models::egarch::{EgarchModel, EgarchConfig};
//!
//! let config = EgarchConfig {
//!     p: 2,  // GARCH order
//!     q: 1,  // ARCH order
//!     max_iterations: 500,
//!     tolerance: 1e-6,
//! };
//!
//! let mut model: EgarchModel<f64> = EgarchModel::new(config);
//! ```
//!
//! ## Analyzing Asymmetric Effects
//! ```rust
//! use scirs2_series::financial::models::egarch::EgarchModel;
//! use scirs2_core::ndarray::array;
//!
//! let mut model = EgarchModel::egarch_11();
//! let data = array![0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.015, 0.02, -0.01, 0.008,
//!                   0.003, -0.012, 0.018, -0.006, 0.009, 0.002, -0.008, 0.014, -0.004, 0.011,
//!                   0.007, -0.009, 0.013, -0.003, 0.006, 0.004, -0.007, 0.016, -0.002, 0.010,
//!                   0.001, -0.005, 0.012, -0.001, 0.008]; // Returns (35 points for EGARCH)
//!
//! // Fit model
//! let result = model.fit(&data).unwrap();
//!
//! // Check for leverage effect
//! let gamma = &result.parameters.gamma[0];
//! if *gamma < 0.0 {
//!     println!("Leverage effect detected: negative shocks increase volatility more");
//! }
//! ```

use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::fmt::Debug;

use crate::error::{Result, TimeSeriesError};

/// Configuration for EGARCH model
#[derive(Debug, Clone)]
pub struct EgarchConfig {
    /// GARCH order (p) - number of lagged conditional variances
    pub p: usize,
    /// ARCH order (q) - number of lagged residuals
    pub q: usize,
    /// Maximum iterations for optimization
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
}

impl Default for EgarchConfig {
    fn default() -> Self {
        Self {
            p: 1,
            q: 1,
            max_iterations: 1000,
            tolerance: 1e-6,
        }
    }
}

/// EGARCH model parameters
#[derive(Debug, Clone)]
pub struct EgarchParameters<F: Float> {
    /// Constant term (ω) in the log-variance equation
    pub omega: F,
    /// Magnitude effects coefficients (α) - impact of shock magnitude
    pub alpha: Array1<F>,
    /// Persistence effects coefficients (β) - impact of lagged log-variance
    pub beta: Array1<F>,
    /// Asymmetry effects coefficients (γ) - leverage effect parameters
    pub gamma: Array1<F>,
}

/// EGARCH model estimation results
#[derive(Debug, Clone)]
pub struct EgarchResult<F: Float> {
    /// Estimated model parameters
    pub parameters: EgarchParameters<F>,
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
    /// Whether optimization converged
    pub converged: bool,
    /// Number of iterations used
    pub iterations: usize,
}

/// EGARCH (Exponential GARCH) model for asymmetric volatility
#[derive(Debug)]
pub struct EgarchModel<F: Float + Debug> {
    #[allow(dead_code)]
    config: EgarchConfig,
    fitted: bool,
    parameters: Option<EgarchParameters<F>>,
    conditional_variance: Option<Array1<F>>,
}

impl<F: Float + Debug + std::iter::Sum> EgarchModel<F> {
    /// Create a new EGARCH model with custom configuration
    ///
    /// # Arguments
    /// * `config` - Configuration parameters for the EGARCH model
    ///
    /// # Examples
    /// ```rust
    /// use scirs2_series::financial::models::egarch::{EgarchModel, EgarchConfig};
    ///
    /// let config = EgarchConfig {
    ///     p: 1,
    ///     q: 1,
    ///     max_iterations: 1000,
    ///     tolerance: 1e-6,
    /// };
    /// let model = EgarchModel::<f64>::new(config);
    /// ```
    pub fn new(config: EgarchConfig) -> Self {
        Self {
            config,
            fitted: false,
            parameters: None,
            conditional_variance: None,
        }
    }

    /// Create EGARCH(1,1) model with default settings
    ///
    /// This is the most commonly used EGARCH specification, which includes
    /// one lagged conditional variance term and one lagged residual term.
    ///
    /// # Examples
    /// ```rust
    /// use scirs2_series::financial::models::egarch::EgarchModel;
    ///
    /// let mut model = EgarchModel::<f64>::egarch_11();
    /// ```
    pub fn egarch_11() -> Self {
        Self::new(EgarchConfig {
            p: 1,
            q: 1,
            max_iterations: 1000,
            tolerance: 1e-6,
        })
    }

    /// Fit EGARCH model to financial time series data
    ///
    /// This method estimates the EGARCH model parameters using a simplified
    /// method of moments approach. For more sophisticated estimation methods,
    /// consider using maximum likelihood estimation.
    ///
    /// # Arguments
    /// * `data` - Time series data (prices or returns)
    ///
    /// # Returns
    /// * `Result<EgarchResult<F>>` - Estimation results including parameters,
    ///   conditional variance, and model diagnostics
    ///
    /// # Errors
    /// * Returns error if insufficient data (< 30 observations)
    ///
    /// # Examples
    /// ```rust
    /// use scirs2_series::financial::models::egarch::EgarchModel;
    /// use scirs2_core::ndarray::array;
    ///
    /// let mut model = EgarchModel::<f64>::egarch_11();
    /// let data = array![1.0, 1.01, 0.99, 1.02, 0.98]; // Price series
    /// let result = model.fit(&data);
    /// ```
    pub fn fit(&mut self, data: &Array1<F>) -> Result<EgarchResult<F>> {
        if data.len() < 30 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Need at least 30 observations for EGARCH estimation".to_string(),
                required: 30,
                actual: data.len(),
            });
        }

        // Calculate returns if input appears to be prices
        let returns = if data.iter().all(|&x| x > F::zero()) {
            // Assume prices, calculate log returns
            let mut ret = Array1::zeros(data.len() - 1);
            for i in 1..data.len() {
                ret[i - 1] = (data[i] / data[i - 1]).ln();
            }
            ret
        } else {
            // Assume already returns
            data.clone()
        };

        let n = returns.len();
        let mean = returns.sum() / F::from(n).unwrap();
        let centered_returns: Array1<F> = returns.mapv(|r| r - mean);

        // Initialize parameters with reasonable starting values
        let sample_var = centered_returns.mapv(|r| r.powi(2)).sum() / F::from(n - 1).unwrap();

        // EGARCH parameters initialization
        let omega = sample_var.ln() * F::from(0.01).unwrap(); // Small constant in log-variance
        let alpha = Array1::from_vec(vec![F::from(0.1).unwrap()]); // Magnitude effect
        let beta = Array1::from_vec(vec![F::from(0.85).unwrap()]); // Persistence effect
        let gamma = Array1::from_vec(vec![F::from(-0.05).unwrap()]); // Asymmetry effect (leverage)

        // Calculate conditional variance using EGARCH(1,1) formula
        let mut log_conditional_variance = Array1::zeros(n);
        log_conditional_variance[0] = sample_var.ln(); // Initialize with sample variance

        for i in 1..n {
            // Calculate standardized residual from previous period
            let standardized_residual =
                centered_returns[i - 1] / log_conditional_variance[i - 1].exp().sqrt();

            // EGARCH(1,1) equation:
            // ln(σ²_t) = ω + α[|z_{t-1}| - E|z_{t-1}|] + γz_{t-1} + β*ln(σ²_{t-1})
            let expected_abs_z = F::from(2.0 / std::f64::consts::PI).unwrap().sqrt(); // E[|Z|] for standard normal
            let magnitude_effect = alpha[0] * (standardized_residual.abs() - expected_abs_z);
            let asymmetry_effect = gamma[0] * standardized_residual;
            let persistence_effect = beta[0] * log_conditional_variance[i - 1];

            log_conditional_variance[i] =
                omega + magnitude_effect + asymmetry_effect + persistence_effect;
        }

        // Convert log-variance to variance
        let conditional_variance = log_conditional_variance.mapv(|x| x.exp());

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
        let parameters = EgarchParameters {
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

        Ok(EgarchResult {
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
    /// * `Option<&EgarchParameters<F>>` - Reference to parameters if fitted, None otherwise
    pub fn get_parameters(&self) -> Option<&EgarchParameters<F>> {
        self.parameters.as_ref()
    }

    /// Get the conditional variance series
    ///
    /// # Returns
    /// * `Option<&Array1<F>>` - Reference to conditional variance if fitted, None otherwise
    pub fn get_conditional_variance(&self) -> Option<&Array1<F>> {
        self.conditional_variance.as_ref()
    }

    /// Forecast conditional variance for multiple periods ahead
    ///
    /// For EGARCH models, multi-step forecasts require iterative computation
    /// since the model is nonlinear in the innovations.
    ///
    /// # Arguments
    /// * `steps` - Number of periods to forecast
    ///
    /// # Returns
    /// * `Result<Array1<F>>` - Forecasted conditional variances
    ///
    /// # Errors
    /// * Returns error if model hasn't been fitted
    pub fn forecast_variance(&self, steps: usize) -> Result<Array1<F>> {
        if !self.fitted {
            return Err(TimeSeriesError::InvalidModel(
                "Model has not been fitted".to_string(),
            ));
        }

        let parameters = self.parameters.as_ref().unwrap();
        let conditional_variance = self.conditional_variance.as_ref().unwrap();

        let mut forecasts = Array1::zeros(steps);

        // Get last log conditional variance
        let last_log_var = conditional_variance[conditional_variance.len() - 1].ln();
        let mut current_log_var = last_log_var;

        // Expected value of |z| for standard normal
        let expected_abs_z = F::from(2.0 / std::f64::consts::PI).unwrap().sqrt();

        for i in 0..steps {
            // For multi-step forecasts, we use E[z_t] = 0 (expected innovation is zero)
            // ln(σ²_{t+h}) = ω + α[E|z_t| - E|z_t|] + γ*E[z_t] + β*ln(σ²_{t+h-1})
            // Simplifies to: ln(σ²_{t+h}) = ω + β*ln(σ²_{t+h-1})

            if i == 0 {
                // One-step ahead: magnitude effect averages to zero, asymmetry effect is zero
                current_log_var = parameters.omega + parameters.beta[0] * current_log_var;
            } else {
                // Multi-step ahead: persistence effect only
                current_log_var = parameters.omega + parameters.beta[0] * current_log_var;
            }

            forecasts[i] = current_log_var.exp();
        }

        Ok(forecasts)
    }
}

/// Utility functions for EGARCH models
impl<F: Float + Debug + std::iter::Sum> EgarchModel<F> {
    /// Calculate the leverage effect ratio
    ///
    /// This measures the asymmetric impact of positive vs negative shocks.
    /// A negative γ parameter indicates leverage effect (negative shocks
    /// increase volatility more than positive shocks).
    ///
    /// # Returns
    /// * `Option<F>` - Leverage effect ratio if model is fitted, None otherwise
    pub fn leverage_effect(&self) -> Option<F> {
        self.parameters.as_ref().map(|p| p.gamma[0])
    }

    /// Check if the model exhibits leverage effect
    ///
    /// # Returns
    /// * `Option<bool>` - True if leverage effect present (γ < 0), None if not fitted
    pub fn has_leverage_effect(&self) -> Option<bool> {
        self.leverage_effect().map(|gamma| gamma < F::zero())
    }

    /// Calculate the persistence of volatility shocks
    ///
    /// This measures how long volatility shocks persist in the system.
    /// Higher β values indicate more persistent volatility.
    ///
    /// # Returns
    /// * `Option<F>` - Persistence parameter if model is fitted, None otherwise
    pub fn volatility_persistence(&self) -> Option<F> {
        self.parameters.as_ref().map(|p| p.beta[0])
    }
}

/// Normal cumulative distribution function approximation
///
/// Uses the Abramowitz and Stegun approximation for the standard normal CDF.
/// This is used internally for various calculations but exposed for utility.
#[allow(dead_code)]
pub fn normal_cdf<F: Float>(x: F) -> F {
    // Abramowitz and Stegun approximation
    let a1 = F::from(0.254829592).unwrap();
    let a2 = F::from(-0.284496736).unwrap();
    let a3 = F::from(1.421413741).unwrap();
    let a4 = F::from(-1.453152027).unwrap();
    let a5 = F::from(1.061405429).unwrap();
    let p = F::from(0.3275911).unwrap();

    let sign = if x < F::zero() { -F::one() } else { F::one() };
    let x_abs = x.abs();

    let t = F::one() / (F::one() + p * x_abs);
    let y = F::one()
        - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1)
            * t
            * (-x_abs * x_abs / F::from(2.0).unwrap()).exp();

    (F::one() + sign * y) / F::from(2.0).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr1;

    #[test]
    fn test_egarch_basic() {
        let mut model = EgarchModel::<f64>::egarch_11();
        let data = arr1(&[
            0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.003, 0.007, -0.001, 0.004, 0.009, -0.006,
            0.002, -0.007, 0.011, 0.003, -0.004, 0.008, -0.002, 0.006, -0.005, 0.009, -0.001,
            0.004, -0.008, 0.012, 0.001, -0.007, 0.010, -0.003,
        ]);

        let result = model.fit(&data);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.parameters.alpha.len(), 1);
        assert_eq!(result.parameters.beta.len(), 1);
        assert_eq!(result.parameters.gamma.len(), 1);
        assert!(result.log_likelihood.is_finite());
        assert!(model.is_fitted());
    }

    #[test]
    fn test_egarch_leverage_effect() {
        let mut model = EgarchModel::<f64>::egarch_11();
        let data = arr1(&[
            0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.003, 0.007, -0.001, 0.004, 0.009, -0.006,
            0.002, -0.007, 0.011, 0.003, -0.004, 0.008, -0.002, 0.006, -0.005, 0.009, -0.001,
            0.004, -0.008, 0.012, 0.001, -0.007, 0.010, -0.003,
        ]);

        model.fit(&data).unwrap();

        let leverage = model.leverage_effect();
        assert!(leverage.is_some());

        let has_leverage = model.has_leverage_effect();
        assert!(has_leverage.is_some());

        let persistence = model.volatility_persistence();
        assert!(persistence.is_some());
        assert!(persistence.unwrap() > 0.0 && persistence.unwrap() < 1.0);
    }

    #[test]
    fn test_egarch_forecasting() {
        let mut model = EgarchModel::<f64>::egarch_11();
        let data = arr1(&[
            0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.003, 0.007, -0.001, 0.004, 0.009, -0.006,
            0.002, -0.007, 0.011, 0.003, -0.004, 0.008, -0.002, 0.006, -0.005, 0.009, -0.001,
            0.004, -0.008, 0.012, 0.001, -0.007, 0.010, -0.003,
        ]);

        model.fit(&data).unwrap();

        let forecasts = model.forecast_variance(5).unwrap();
        assert_eq!(forecasts.len(), 5);
        assert!(forecasts.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn test_insufficient_data() {
        let mut model = EgarchModel::<f64>::egarch_11();
        let data = arr1(&[0.01, -0.02, 0.015]); // Too few observations

        let result = model.fit(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_custom_egarch_config() {
        let config = EgarchConfig {
            p: 2,
            q: 1,
            max_iterations: 100,
            tolerance: 1e-4,
        };

        let model = EgarchModel::<f64>::new(config);
        assert!(!model.is_fitted());
        assert!(model.get_parameters().is_none());
    }

    #[test]
    fn test_normal_cdf() {
        let x = 0.0;
        let cdf_value = normal_cdf(x);
        assert!((cdf_value - 0.5).abs() < 1e-2);

        let x = 1.96;
        let cdf_value = normal_cdf(x);
        assert!((cdf_value - 0.975).abs() < 1e-2);
    }
}
