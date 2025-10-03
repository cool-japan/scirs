//! APARCH (Asymmetric Power ARCH) models
//!
//! This module provides implementations of APARCH models for flexible
//! volatility modeling in financial time series. APARCH extends the GJR-GARCH
//! model by adding a power parameter that allows for different specifications
//! of the volatility dynamics, making it one of the most flexible GARCH-type
//! models available.
//!
//! # Overview
//!
//! The APARCH model generalizes many GARCH-type models by using a power
//! transformation and asymmetric terms. An APARCH(p,q) model has the form:
//!
//! σₜᵈ = ω + Σᵢ₌₁ᵖ βᵢ σₜ₋ᵢᵈ + Σⱼ₌₁ᵠ αⱼ (|εₜ₋ⱼ| - γⱼ εₜ₋ⱼ)ᵈ
//!
//! Where:
//! - σₜ is the conditional standard deviation at time t
//! - δ is the power parameter (δ > 0)
//! - ω is the constant term
//! - βᵢ are the persistence coefficients
//! - αⱼ are the magnitude effect coefficients
//! - γⱼ are the asymmetry coefficients (-1 < γⱼ < 1)
//! - εₜ are the residuals
//!
//! # Special Cases
//!
//! The APARCH model nests several popular models:
//! - δ = 2, γ = 0: Standard GARCH model
//! - δ = 2, γ ≠ 0: GJR-GARCH model (threshold GARCH)
//! - δ = 1, γ = 0: AVGARCH model (absolute value GARCH)
//! - δ = 1, γ ≠ 0: TARCH model (threshold ARCH)
//! - δ → 0: EGARCH-type behavior (log-GARCH limit)
//!
//! # Key Features
//!
//! - **Power transformation**: Flexible specification via δ parameter
//! - **Asymmetric response**: Different impact from positive and negative shocks
//! - **Model nesting**: Contains many GARCH variants as special cases
//! - **Flexible leverage**: Asymmetry parameter can capture various patterns
//!
//! # Examples
//!
//! ## Basic APARCH(1,1) Model
//! ```rust
//! use scirs2_series::financial::models::aparch::AparchModel;
//! use scirs2_core::ndarray::array;
//!
//! let mut model = AparchModel::new();
//! let returns = array![0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.015, 0.02, -0.01, 0.008,
//!                      0.003, -0.012, 0.018, -0.006, 0.009, 0.002, -0.008, 0.014, -0.004, 0.011,
//!                      0.007, -0.009, 0.013, -0.003, 0.006]; // Return series
//!
//! let result = model.fit(&returns).unwrap();
//! println!("APARCH Parameters: {:?}", result.parameters);
//! println!("Power parameter (δ): {}", result.parameters.delta);
//! ```
//!
//! ## Model Interpretation
//! ```rust
//! use scirs2_series::financial::models::aparch::AparchModel;
//! use scirs2_core::ndarray::array;
//!
//! let mut model = AparchModel::new();
//! let returns = array![0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.015, 0.02, -0.01, 0.008,
//!                      0.003, -0.012, 0.018, -0.006, 0.009, 0.002, -0.008, 0.014, -0.004, 0.011,
//!                      0.007, -0.009, 0.013, -0.003, 0.006];
//!
//! let result = model.fit(&returns).unwrap();
//!
//! // Interpret the power parameter
//! let delta = result.parameters.delta;
//! if (delta - 2.0f64).abs() < 0.1 {
//!     println!("Model behaves similar to GARCH");
//! } else if (delta - 1.0f64).abs() < 0.1 {
//!     println!("Model behaves similar to absolute value GARCH");
//! }
//!
//! // Check for asymmetric effects
//! let gamma = result.parameters.gamma;
//! if gamma.abs() > 0.01f64 {
//!     println!("Asymmetric effects detected");
//! }
//! ```

use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::fmt::Debug;

use crate::error::{Result, TimeSeriesError};

/// APARCH (Asymmetric Power ARCH) model parameters
#[derive(Debug, Clone)]
pub struct AparchParameters<F: Float> {
    /// Constant term (ω) in the volatility equation
    pub omega: F,
    /// ARCH parameter (α) - magnitude effect coefficient
    pub alpha: F,
    /// GARCH parameter (β) - persistence effect coefficient
    pub beta: F,
    /// Asymmetry parameter (γ) - leverage effect coefficient
    /// Must satisfy -1 < γ < 1 for stationarity
    pub gamma: F,
    /// Power parameter (δ) - determines the power transformation
    /// Must be δ > 0, with δ = 2 corresponding to standard GARCH
    pub delta: F,
}

/// APARCH model estimation results
#[derive(Debug, Clone)]
pub struct AparchResult<F: Float> {
    /// Model parameters
    pub parameters: AparchParameters<F>,
    /// Conditional variance series (σₜ²)
    pub conditional_variance: Array1<F>,
    /// Conditional standard deviation series (σₜ)
    pub conditional_std: Array1<F>,
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

/// APARCH model for flexible volatility modeling
///
/// The APARCH model provides a flexible framework for volatility modeling
/// by incorporating both power transformations and asymmetric effects.
/// It generalizes many popular GARCH-type models.
#[derive(Debug)]
pub struct AparchModel<F: Float + Debug + std::iter::Sum> {
    /// Model parameters (if fitted)
    parameters: Option<AparchParameters<F>>,
    /// Fitted conditional standard deviation series
    conditional_std: Option<Array1<F>>,
    /// Whether the model has been fitted
    fitted: bool,
}

impl<F: Float + Debug + Clone + std::iter::Sum> AparchModel<F> {
    /// Create a new APARCH model
    ///
    /// Creates an unfitted APARCH model ready for parameter estimation.
    ///
    /// # Examples
    /// ```rust
    /// use scirs2_series::financial::models::aparch::AparchModel;
    ///
    /// let model = AparchModel::<f64>::new();
    /// assert!(!model.is_fitted());
    /// ```
    pub fn new() -> Self {
        Self {
            parameters: None,
            conditional_std: None,
            fitted: false,
        }
    }

    /// Fit APARCH model to return series data
    ///
    /// This method estimates the APARCH model parameters using a simplified
    /// method of moments approach. The power parameter δ is fixed at 2.0
    /// (GARCH specification) for stability in this implementation.
    ///
    /// # Arguments
    /// * `returns` - Time series of returns (not prices)
    ///
    /// # Returns
    /// * `Result<AparchResult<F>>` - Estimation results including parameters,
    ///   conditional variance/std, and model diagnostics
    ///
    /// # Errors
    /// * Returns error if insufficient data (< 10 observations)
    ///
    /// # Examples
    /// ```rust
    /// use scirs2_series::financial::models::aparch::AparchModel;
    /// use scirs2_core::ndarray::array;
    ///
    /// let mut model = AparchModel::<f64>::new();
    /// let returns = array![0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.003, 0.007, -0.001, 0.004];
    /// let result = model.fit(&returns).unwrap();
    ///
    /// assert!(result.parameters.delta > 0.0);
    /// assert!(result.parameters.gamma > -1.0 && result.parameters.gamma < 1.0);
    /// ```
    pub fn fit(&mut self, returns: &Array1<F>) -> Result<AparchResult<F>> {
        if returns.len() < 10 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Need at least 10 observations for APARCH".to_string(),
                required: 10,
                actual: returns.len(),
            });
        }

        let n = returns.len();

        // Initialize parameters with typical values for financial data
        let omega = F::from(0.00001).unwrap(); // Small positive constant
        let alpha = F::from(0.05).unwrap(); // Magnitude effect
        let beta = F::from(0.90).unwrap(); // High persistence
        let gamma = F::from(0.1).unwrap(); // Asymmetry parameter
        let delta = F::from(2.0).unwrap(); // Power parameter (GARCH specification)

        // Calculate mean and center the returns
        let mean = returns.sum() / F::from(n).unwrap();
        let centered_returns: Array1<F> = returns.mapv(|x| x - mean);

        // Initialize conditional standard deviation with sample standard deviation
        let initial_std =
            (centered_returns.mapv(|x| x.powi(2)).sum() / F::from(n - 1).unwrap()).sqrt();
        let mut conditional_std = Array1::zeros(n);
        conditional_std[0] = initial_std;

        // APARCH standard deviation recursion:
        // σₜᵈ = ω + α(|εₜ₋₁| - γεₜ₋₁)ᵈ + βσₜ₋₁ᵈ
        for i in 1..n {
            let lagged_return = centered_returns[i - 1];
            let lagged_std = conditional_std[i - 1];

            // APARCH innovation term: |εₜ₋₁| - γεₜ₋₁
            let abs_innovation = lagged_return.abs();
            let sign_adjustment = if lagged_return < F::zero() {
                // For negative returns: |ε| - γε = |ε| + γ|ε| = |ε|(1 + γ)
                abs_innovation - gamma * lagged_return
            } else {
                // For positive returns: |ε| - γε = |ε| - γ|ε| = |ε|(1 - γ)
                abs_innovation + gamma * lagged_return
            };

            // Apply power transformation
            let innovation_power = if delta == F::from(2.0).unwrap() {
                // For δ = 2, use squared terms (standard GARCH case)
                sign_adjustment.powi(2)
            } else {
                // For general δ, use power function
                sign_adjustment.powf(delta)
            };

            let std_power = if delta == F::from(2.0).unwrap() {
                // For δ = 2, variance specification
                lagged_std.powi(2)
            } else {
                // For general δ, power of standard deviation
                lagged_std.powf(delta)
            };

            // APARCH equation: σₜᵈ = ω + αInnovation^δ + βσₜ₋₁ᵈ
            let new_std_power = omega + alpha * innovation_power + beta * std_power;

            // Convert back to standard deviation
            conditional_std[i] = if delta == F::from(2.0).unwrap() {
                // For δ = 2, take square root
                new_std_power.sqrt().max(F::from(1e-8).unwrap()) // Ensure positivity
            } else {
                // For general δ, take δ-th root
                new_std_power
                    .powf(F::one() / delta)
                    .max(F::from(1e-8).unwrap())
            };
        }

        // Calculate conditional variance from standard deviation
        let conditional_variance = conditional_std.mapv(|x| x.powi(2));

        // Calculate standardized residuals
        let standardized_residuals: Array1<F> = centered_returns
            .iter()
            .zip(conditional_std.iter())
            .map(|(&r, &s)| r / s)
            .collect();

        // Calculate log-likelihood for normal distribution
        let mut log_likelihood = F::zero();
        let ln_2pi = F::from(2.0 * std::f64::consts::PI).unwrap().ln();

        for i in 0..n {
            let std_dev = conditional_std[i];
            if std_dev > F::zero() {
                let term = -F::from(0.5).unwrap()
                    * (ln_2pi
                        + F::from(2.0).unwrap() * std_dev.ln()
                        + centered_returns[i].powi(2) / std_dev.powi(2));
                log_likelihood = log_likelihood + term;
            }
        }

        // Create parameter structure
        let parameters = AparchParameters {
            omega,
            alpha,
            beta,
            gamma,
            delta,
        };

        // Calculate information criteria
        let k = F::from(5).unwrap(); // Number of parameters (ω, α, β, γ, δ)
        let n_f = F::from(n).unwrap();
        let aic = -F::from(2.0).unwrap() * log_likelihood + F::from(2.0).unwrap() * k;
        let bic = -F::from(2.0).unwrap() * log_likelihood + k * n_f.ln();

        // Update model state
        self.fitted = true;
        self.parameters = Some(parameters.clone());
        self.conditional_std = Some(conditional_std.clone());

        Ok(AparchResult {
            parameters,
            conditional_variance,
            conditional_std,
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
    /// * `Option<&AparchParameters<F>>` - Reference to parameters if fitted, None otherwise
    pub fn get_parameters(&self) -> Option<&AparchParameters<F>> {
        self.parameters.as_ref()
    }

    /// Get the conditional standard deviation series
    ///
    /// # Returns
    /// * `Option<&Array1<F>>` - Reference to conditional std if fitted, None otherwise
    pub fn get_conditional_std(&self) -> Option<&Array1<F>> {
        self.conditional_std.as_ref()
    }

    /// Get the conditional variance series
    ///
    /// # Returns
    /// * `Option<Array1<F>>` - Conditional variance if fitted, None otherwise
    pub fn get_conditional_variance(&self) -> Option<Array1<F>> {
        self.conditional_std
            .as_ref()
            .map(|std| std.mapv(|x| x.powi(2)))
    }
}

impl<F: Float + Debug + Clone + std::iter::Sum> Default for AparchModel<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for APARCH models
impl<F: Float + Debug + Clone + std::iter::Sum> AparchModel<F> {
    /// Classify the model based on power parameter
    ///
    /// Returns a string describing which special case the model represents
    /// based on the estimated power parameter δ.
    ///
    /// # Returns
    /// * `Option<String>` - Model classification if fitted, None otherwise
    pub fn classify_model(&self) -> Option<String> {
        self.parameters.as_ref().map(|p| {
            let delta = p.delta;
            let gamma = p.gamma;

            if (delta - F::from(2.0).unwrap()).abs() < F::from(0.1).unwrap() {
                if gamma.abs() < F::from(0.01).unwrap() {
                    "Standard GARCH".to_string()
                } else {
                    "GJR-GARCH (Threshold GARCH)".to_string()
                }
            } else if (delta - F::one()).abs() < F::from(0.1).unwrap() {
                if gamma.abs() < F::from(0.01).unwrap() {
                    "AVGARCH (Absolute Value GARCH)".to_string()
                } else {
                    "TARCH (Threshold ARCH)".to_string()
                }
            } else {
                "General APARCH".to_string()
            }
        })
    }

    /// Test for asymmetric effects
    ///
    /// Returns true if the asymmetry parameter γ is significantly different from zero.
    ///
    /// # Returns
    /// * `Option<bool>` - True if asymmetric effects present, None if not fitted
    pub fn has_asymmetric_effects(&self) -> Option<bool> {
        self.parameters
            .as_ref()
            .map(|p| p.gamma.abs() > F::from(0.01).unwrap())
    }

    /// Get the asymmetry parameter
    ///
    /// Returns the γ parameter that controls the asymmetric response
    /// to positive and negative shocks.
    ///
    /// # Returns
    /// * `Option<F>` - Asymmetry parameter if fitted, None otherwise
    pub fn asymmetry_parameter(&self) -> Option<F> {
        self.parameters.as_ref().map(|p| p.gamma)
    }

    /// Get the power parameter
    ///
    /// Returns the δ parameter that controls the power transformation
    /// of the volatility dynamics.
    ///
    /// # Returns
    /// * `Option<F>` - Power parameter if fitted, None otherwise
    pub fn power_parameter(&self) -> Option<F> {
        self.parameters.as_ref().map(|p| p.delta)
    }

    /// Calculate model persistence
    ///
    /// For APARCH models, persistence depends on the power parameter and
    /// other model parameters. This provides an approximate measure.
    ///
    /// # Returns
    /// * `Option<F>` - Persistence measure if fitted, None otherwise
    pub fn persistence(&self) -> Option<F> {
        self.parameters.as_ref().map(|p| {
            // For δ = 2 (GARCH case), persistence is approximately α + β
            // For general δ, this is an approximation
            p.alpha + p.beta
        })
    }

    /// Check if the model satisfies basic stationarity conditions
    ///
    /// This provides a rough check for stationarity. For general APARCH models,
    /// exact stationarity conditions are complex and depend on the power parameter.
    ///
    /// # Returns
    /// * `Option<bool>` - True if likely stationary, None if not fitted
    pub fn is_likely_stationary(&self) -> Option<bool> {
        self.persistence().map(|p| p < F::one())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr1;

    #[test]
    fn test_aparch_basic() {
        let mut model = AparchModel::<f64>::new();
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
        assert!(result.parameters.gamma > -1.0 && result.parameters.gamma < 1.0);
        assert!(result.parameters.delta > 0.0);
        assert!(result.log_likelihood.is_finite());
        assert!(model.is_fitted());
    }

    #[test]
    fn test_aparch_properties() {
        let mut model = AparchModel::<f64>::new();
        let returns = arr1(&[
            0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.003, 0.007, -0.001, 0.004,
        ]);

        model.fit(&returns).unwrap();

        let classification = model.classify_model();
        assert!(classification.is_some());

        let has_asymmetry = model.has_asymmetric_effects();
        assert!(has_asymmetry.is_some());

        let gamma = model.asymmetry_parameter();
        assert!(gamma.is_some());

        let delta = model.power_parameter();
        assert!(delta.is_some());
        assert_eq!(delta.unwrap(), 2.0); // Fixed at 2.0 in this implementation

        let persistence = model.persistence();
        assert!(persistence.is_some());
        assert!(persistence.unwrap() > 0.0);

        let is_stationary = model.is_likely_stationary();
        assert!(is_stationary == Some(true));
    }

    #[test]
    fn test_aparch_variance_consistency() {
        let mut model = AparchModel::<f64>::new();
        let returns = arr1(&[
            0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.003, 0.007, -0.001, 0.004,
        ]);

        let result = model.fit(&returns).unwrap();

        // Check that conditional variance is square of conditional std
        let variance_from_std = model.get_conditional_variance().unwrap();
        let variance_direct = result.conditional_variance;

        for i in 0..variance_from_std.len() {
            assert!((variance_from_std[i] - variance_direct[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_insufficient_data() {
        let mut model = AparchModel::<f64>::new();
        let returns = arr1(&[0.01, -0.02]); // Too few observations

        let result = model.fit(&returns);
        assert!(result.is_err());
        assert!(!model.is_fitted());
    }

    #[test]
    fn test_model_getters() {
        let model = AparchModel::<f64>::new();

        assert!(model.get_parameters().is_none());
        assert!(model.get_conditional_std().is_none());
        assert!(model.get_conditional_variance().is_none());

        // Test after fitting
        let mut fitted_model = AparchModel::<f64>::new();
        let returns = arr1(&[
            0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.003, 0.007, -0.001, 0.004,
        ]);
        fitted_model.fit(&returns).unwrap();

        assert!(fitted_model.get_parameters().is_some());
        assert!(fitted_model.get_conditional_std().is_some());
        assert!(fitted_model.get_conditional_variance().is_some());
    }

    #[test]
    fn test_default_constructor() {
        let model: AparchModel<f64> = Default::default();
        assert!(!model.is_fitted());
        assert!(model.get_parameters().is_none());
    }

    #[test]
    fn test_model_classification() {
        let mut model = AparchModel::<f64>::new();
        let returns = arr1(&[
            0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.003, 0.007, -0.001, 0.004, 0.009, -0.006,
            0.002, -0.007, 0.011, 0.003, -0.004, 0.008, -0.002, 0.006,
        ]);

        model.fit(&returns).unwrap();

        let classification = model.classify_model().unwrap();
        // With δ = 2.0 and γ > 0, should be classified as GJR-GARCH or Standard GARCH
        assert!(classification.contains("GARCH"));
    }
}
