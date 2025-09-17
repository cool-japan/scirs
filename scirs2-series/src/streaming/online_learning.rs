//! Online learning algorithms for streaming time series analysis
//!
//! This module provides adaptive machine learning algorithms that can learn
//! from streaming data, including adaptive linear regression and ARIMA models.

use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive};
use std::collections::VecDeque;
use std::fmt::Debug;

use crate::error::{Result, TimeSeriesError};

/// Adaptive linear regression with forgetting factor
#[derive(Debug)]
pub struct AdaptiveLinearRegression<F: Float + Debug> {
    /// Regression coefficients
    coefficients: Array1<F>,
    /// Covariance matrix
    covariance: Array2<F>,
    /// Forgetting factor (0 < lambda <= 1)
    forgetting_factor: F,
    /// Regularization parameter
    regularization: F,
    /// Number of features
    num_features: usize,
    /// Update counter
    update_count: usize,
}

impl<F: Float + Debug + Clone + FromPrimitive> AdaptiveLinearRegression<F> {
    /// Create new adaptive linear regression
    pub fn new(_num_features: usize, forgettingfactor: F, regularization: F) -> Result<Self> {
        if forgettingfactor <= F::zero() || forgettingfactor > F::one() {
            return Err(TimeSeriesError::InvalidParameter {
                name: "forgetting_factor".to_string(),
                message: "Forgetting _factor must be in (0, 1]".to_string(),
            });
        }

        let mut covariance = Array2::zeros((_num_features, _num_features));
        let identity_scale = F::from(1000.0).unwrap(); // Large initial uncertainty
        for i in 0.._num_features {
            covariance[[i, i]] = identity_scale;
        }

        Ok(Self {
            coefficients: Array1::zeros(_num_features),
            covariance,
            forgetting_factor: forgettingfactor,
            regularization,
            num_features: _num_features,
            update_count: 0,
        })
    }

    /// Update model with new observation using Recursive Least Squares
    pub fn update(&mut self, features: &Array1<F>, target: F) -> Result<()> {
        if features.len() != self.num_features {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: self.num_features,
                actual: features.len(),
            });
        }

        self.update_count += 1;

        // Compute prediction error
        let prediction = self.predict(features)?;
        let error = target - prediction;

        // RLS update
        let mut temp_vector = Array1::zeros(self.num_features);
        for i in 0..self.num_features {
            let mut sum = F::zero();
            for j in 0..self.num_features {
                sum = sum + self.covariance[[i, j]] * features[j];
            }
            temp_vector[i] = sum;
        }

        let mut denominator = self.forgetting_factor;
        for i in 0..self.num_features {
            denominator = denominator + features[i] * temp_vector[i];
        }

        // Kalman gain
        let mut gain = Array1::zeros(self.num_features);
        for i in 0..self.num_features {
            gain[i] = temp_vector[i] / denominator;
        }

        // Update coefficients
        for i in 0..self.num_features {
            self.coefficients[i] = self.coefficients[i] + gain[i] * error;
        }

        // Update covariance matrix
        let mut new_covariance = Array2::zeros((self.num_features, self.num_features));
        for i in 0..self.num_features {
            for j in 0..self.num_features {
                let _update_term = gain[i] * features[j];
                new_covariance[[i, j]] = (self.covariance[[i, j]] - temp_vector[i] * features[j])
                    / self.forgetting_factor;

                // Add regularization
                if i == j {
                    new_covariance[[i, j]] = new_covariance[[i, j]] + self.regularization;
                }
            }
        }
        self.covariance = new_covariance;

        Ok(())
    }

    /// Make prediction
    pub fn predict(&self, features: &Array1<F>) -> Result<F> {
        if features.len() != self.num_features {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: self.num_features,
                actual: features.len(),
            });
        }

        let mut prediction = F::zero();
        for i in 0..self.num_features {
            prediction = prediction + self.coefficients[i] * features[i];
        }

        Ok(prediction)
    }

    /// Get prediction with confidence interval
    pub fn predict_with_confidence(
        &self,
        features: &Array1<F>,
        _confidence_level: F,
    ) -> Result<(F, F, F)> {
        let prediction = self.predict(features)?;

        // Compute prediction variance
        let mut variance = F::zero();
        for i in 0..self.num_features {
            for j in 0..self.num_features {
                variance = variance + features[i] * self.covariance[[i, j]] * features[j];
            }
        }

        let std_dev = variance.sqrt();
        let z_score = F::from(1.96).unwrap(); // 95% confidence interval

        let lower_bound = prediction - z_score * std_dev;
        let upper_bound = prediction + z_score * std_dev;

        Ok((prediction, lower_bound, upper_bound))
    }

    /// Get current coefficients
    pub fn get_coefficients(&self) -> &Array1<F> {
        &self.coefficients
    }

    /// Get model confidence (trace of covariance matrix)
    pub fn get_confidence(&self) -> F {
        let mut trace = F::zero();
        for i in 0..self.num_features {
            trace = trace + self.covariance[[i, i]];
        }
        trace / F::from(self.num_features).unwrap()
    }
}

/// Adaptive ARIMA model with online parameter estimation
#[derive(Debug)]
pub struct AdaptiveARIMA<F: Float + Debug> {
    /// AR order
    p: usize,
    /// Differencing order
    d: usize,
    /// MA order
    q: usize,
    /// AR coefficients
    ar_coeffs: Array1<F>,
    /// MA coefficients
    ma_coeffs: Array1<F>,
    /// Recent observations
    observations: VecDeque<F>,
    /// Recent residuals
    residuals: VecDeque<F>,
    /// Learning rate
    learning_rate: F,
    /// Model has been initialized
    initialized: bool,
}

impl<F: Float + Debug + Clone + FromPrimitive> AdaptiveARIMA<F> {
    /// Create new adaptive ARIMA model
    pub fn new(p: usize, d: usize, q: usize, learningrate: F) -> Self {
        Self {
            p,
            d,
            q,
            ar_coeffs: Array1::zeros(p),
            ma_coeffs: Array1::zeros(q),
            observations: VecDeque::with_capacity(100),
            residuals: VecDeque::with_capacity(100),
            learning_rate: learningrate,
            initialized: false,
        }
    }

    /// Update model with new observation
    pub fn update(&mut self, observation: F) -> Result<()> {
        // Add observation to buffer
        if self.observations.len() >= 100 {
            self.observations.pop_front();
        }
        self.observations.push_back(observation);

        // Apply differencing if needed
        let processed_obs = if self.d > 0 && self.observations.len() > self.d {
            self.apply_differencing(observation)
        } else {
            observation
        };

        // Initialize model if we have enough data
        if !self.initialized && self.observations.len() >= self.p.max(self.q) + self.d + 10 {
            self.initialize_parameters()?;
            self.initialized = true;
        }

        if self.initialized {
            // Make prediction and compute residual
            let prediction = self.predict_next()?;
            let residual = processed_obs - prediction;

            // Update residuals buffer
            if self.residuals.len() >= 100 {
                self.residuals.pop_front();
            }
            self.residuals.push_back(residual);

            // Update parameters using gradient descent
            self.update_parameters(processed_obs, residual)?;
        }

        Ok(())
    }

    /// Apply differencing to observation
    fn apply_differencing(&self, observation: F) -> F {
        let len = self.observations.len();
        if len <= self.d {
            return observation;
        }

        let mut diff_obs = observation;
        for _ in 0..self.d {
            diff_obs = diff_obs - self.observations[len - 1];
        }
        diff_obs
    }

    /// Initialize parameters using method of moments
    fn initialize_parameters(&mut self) -> Result<()> {
        let processed_data: Vec<F> = if self.d > 0 {
            self.apply_differencing_to_series()
        } else {
            self.observations.iter().cloned().collect()
        };

        if processed_data.len() < self.p.max(self.q) + 5 {
            return Ok(());
        }

        // Simple initialization using autocorrelations
        self.initialize_ar_parameters(&processed_data)?;
        self.initialize_ma_parameters(&processed_data)?;

        Ok(())
    }

    /// Apply differencing to entire series
    fn apply_differencing_to_series(&self) -> Vec<F> {
        let mut series: Vec<F> = self.observations.iter().cloned().collect();

        for _ in 0..self.d {
            let mut diff_series = Vec::new();
            for i in 1..series.len() {
                diff_series.push(series[i] - series[i - 1]);
            }
            series = diff_series;
        }

        series
    }

    /// Initialize AR parameters using Yule-Walker equations
    fn initialize_ar_parameters(&mut self, data: &[F]) -> Result<()> {
        if self.p == 0 || data.len() < self.p + 1 {
            return Ok(());
        }

        // Compute autocorrelations
        let mut autocorrs = vec![F::zero(); self.p + 1];
        for lag in 0..=self.p {
            let mut sum = F::zero();
            let mut count = 0;

            for i in lag..data.len() {
                sum = sum + data[i] * data[i - lag];
                count += 1;
            }

            if count > 0 {
                autocorrs[lag] = sum / F::from(count).unwrap();
            }
        }

        // Solve Yule-Walker equations (simplified)
        for i in 0..self.p {
            let mut coeff = F::zero();
            if autocorrs[0] > F::zero() {
                coeff = autocorrs[i + 1] / autocorrs[0];
            }
            self.ar_coeffs[i] = coeff
                .max(F::from(-0.99).unwrap())
                .min(F::from(0.99).unwrap());
        }

        Ok(())
    }

    /// Initialize MA parameters (simplified)
    fn initialize_ma_parameters(&mut self, data: &[F]) -> Result<()> {
        // Simple initialization: small random values
        for i in 0..self.q {
            self.ma_coeffs[i] = F::from(0.1).unwrap() * F::from((i + 1) as f64 * 0.1).unwrap();
        }
        Ok(())
    }

    /// Predict next value
    fn predict_next(&self) -> Result<F> {
        if !self.initialized {
            return Ok(F::zero());
        }

        let processed_data: Vec<F> = if self.d > 0 {
            self.apply_differencing_to_series()
        } else {
            self.observations.iter().cloned().collect()
        };

        let mut prediction = F::zero();

        // AR component
        for i in 0..self.p {
            if i < processed_data.len() {
                let lag_index = processed_data.len() - 1 - i;
                prediction = prediction + self.ar_coeffs[i] * processed_data[lag_index];
            }
        }

        // MA component
        for i in 0..self.q {
            if i < self.residuals.len() {
                let lag_index = self.residuals.len() - 1 - i;
                prediction = prediction + self.ma_coeffs[i] * self.residuals[lag_index];
            }
        }

        Ok(prediction)
    }

    /// Update parameters using gradient descent
    fn update_parameters(&mut self, observation: F, residual: F) -> Result<()> {
        let processed_data: Vec<F> = if self.d > 0 {
            self.apply_differencing_to_series()
        } else {
            self.observations.iter().cloned().collect()
        };

        // Update AR coefficients
        for i in 0..self.p {
            if i < processed_data.len() {
                let lag_index = processed_data.len() - 1 - i;
                let gradient = residual * processed_data[lag_index];
                self.ar_coeffs[i] = self.ar_coeffs[i] + self.learning_rate * gradient;

                // Keep coefficients stable
                self.ar_coeffs[i] = self.ar_coeffs[i]
                    .max(F::from(-0.99).unwrap())
                    .min(F::from(0.99).unwrap());
            }
        }

        // Update MA coefficients
        for i in 0..self.q {
            if i < self.residuals.len() {
                let lag_index = self.residuals.len() - 1 - i;
                let gradient = residual * self.residuals[lag_index];
                self.ma_coeffs[i] = self.ma_coeffs[i] + self.learning_rate * gradient;

                // Keep coefficients stable
                self.ma_coeffs[i] = self.ma_coeffs[i]
                    .max(F::from(-0.99).unwrap())
                    .min(F::from(0.99).unwrap());
            }
        }

        Ok(())
    }

    /// Generate forecast
    pub fn forecast(&self, steps: usize) -> Result<Array1<F>> {
        if !self.initialized {
            return Ok(Array1::zeros(steps));
        }

        let mut forecasts = Array1::zeros(steps);
        let mut extended_data = self.observations.clone();
        let mut extended_residuals = self.residuals.clone();

        for step in 0..steps {
            // Apply differencing to get stationary series
            let processed_data: Vec<F> = if self.d > 0 {
                self.apply_differencing_to_extended(&extended_data)
            } else {
                extended_data.iter().cloned().collect()
            };

            let mut prediction = F::zero();

            // AR component
            for i in 0..self.p {
                if i < processed_data.len() {
                    let lag_index = processed_data.len() - 1 - i;
                    prediction = prediction + self.ar_coeffs[i] * processed_data[lag_index];
                }
            }

            // MA component
            for i in 0..self.q {
                if i < extended_residuals.len() {
                    let lag_index = extended_residuals.len() - 1 - i;
                    prediction = prediction + self.ma_coeffs[i] * extended_residuals[lag_index];
                }
            }

            // Convert back from differenced space if needed
            let forecast = if self.d > 0 && !extended_data.is_empty() {
                prediction + extended_data[extended_data.len() - 1]
            } else {
                prediction
            };

            forecasts[step] = forecast;

            // Extend data for next iteration
            extended_data.push_back(forecast);
            extended_residuals.push_back(F::zero()); // Assume zero residual for future

            // Maintain buffer size
            if extended_data.len() > 100 {
                extended_data.pop_front();
            }
            if extended_residuals.len() > 100 {
                extended_residuals.pop_front();
            }
        }

        Ok(forecasts)
    }

    /// Apply differencing to extended data
    fn apply_differencing_to_extended(&self, data: &VecDeque<F>) -> Vec<F> {
        let mut series: Vec<F> = data.iter().cloned().collect();

        for _ in 0..self.d {
            let mut diff_series = Vec::new();
            for i in 1..series.len() {
                diff_series.push(series[i] - series[i - 1]);
            }
            series = diff_series;
        }

        series
    }

    /// Get current model parameters
    pub fn get_parameters(&self) -> (Array1<F>, Array1<F>) {
        (self.ar_coeffs.clone(), self.ma_coeffs.clone())
    }

    /// Check if model is initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
}
