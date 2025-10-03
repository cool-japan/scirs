//! Advanced extrapolation functionality
//!
//! This module contains the AdvancedExtrapolator struct and its implementation
//! for performing sophisticated extrapolation using ensemble, adaptive, and
//! statistical methods.

use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::{Float, FromPrimitive};
use std::default::Default;
use std::ops::AddAssign;

use crate::error::{InterpolateError, InterpolateResult};

use super::config::{
    AdaptiveExtrapolationConfig, AutoregressiveExtrapolationConfig, ConfidenceExtrapolationConfig,
    ConfidenceExtrapolationResult, EnsembleExtrapolationConfig,
};
use super::core::Extrapolator;
use super::types::{ARFittingMethod, EnsembleCombinationStrategy, ExtrapolationMethod};

/// Advanced extrapolator with ensemble, adaptive, and statistical capabilities
#[derive(Debug, Clone)]
pub struct AdvancedExtrapolator<T: Float> {
    /// Basic extrapolator for standard methods
    pub base_extrapolator: Extrapolator<T>,
    /// Configuration for confidence-based extrapolation
    pub confidence_config: Option<ConfidenceExtrapolationConfig<T>>,
    /// Configuration for ensemble extrapolation
    pub ensemble_config: Option<EnsembleExtrapolationConfig<T>>,
    /// Configuration for adaptive extrapolation
    pub adaptive_config: Option<AdaptiveExtrapolationConfig>,
    /// Configuration for autoregressive extrapolation
    pub autoregressive_config: Option<AutoregressiveExtrapolationConfig<T>>,
    /// Historical data for advanced methods (when available)
    pub historical_data: Option<(Array1<T>, Array1<T>)>,
}

impl<T: Float + std::fmt::Display + Default + AddAssign> AdvancedExtrapolator<T> {
    /// Create a new advanced extrapolator
    pub fn new(base_extrapolator: Extrapolator<T>) -> Self {
        Self {
            base_extrapolator,
            confidence_config: None,
            ensemble_config: None,
            adaptive_config: None,
            autoregressive_config: None,
            historical_data: None,
        }
    }

    /// Enable confidence-based extrapolation
    pub fn with_confidence(mut self, config: ConfidenceExtrapolationConfig<T>) -> Self {
        self.confidence_config = Some(config);
        self
    }

    /// Enable ensemble extrapolation
    pub fn with_ensemble(mut self, config: EnsembleExtrapolationConfig<T>) -> Self {
        self.ensemble_config = Some(config);
        self
    }

    /// Enable adaptive extrapolation
    pub fn with_adaptive(mut self, config: AdaptiveExtrapolationConfig) -> Self {
        self.adaptive_config = Some(config);
        self
    }

    /// Enable autoregressive extrapolation
    pub fn with_autoregressive(mut self, config: AutoregressiveExtrapolationConfig<T>) -> Self {
        self.autoregressive_config = Some(config);
        self
    }

    /// Set historical data for advanced methods
    pub fn with_historical_data(mut self, x_data: Array1<T>, y_data: Array1<T>) -> Self {
        self.historical_data = Some((x_data, y_data));
        self
    }

    /// Perform advanced extrapolation at a point
    pub fn extrapolate_advanced(&self, x: T) -> InterpolateResult<T> {
        // Try ensemble extrapolation first if configured
        if self.ensemble_config.is_some() {
            return self.extrapolate_ensemble(x);
        }

        // Try adaptive extrapolation if configured
        if self.adaptive_config.is_some() {
            return self.extrapolate_adaptive(x);
        }

        // Try autoregressive extrapolation if configured
        if self.autoregressive_config.is_some() {
            return self.extrapolate_autoregressive(x);
        }

        // Fall back to base extrapolator
        self.base_extrapolator.extrapolate(x)
    }

    /// Perform confidence-based extrapolation
    pub fn extrapolate_with_confidence(
        &self,
        x: T,
    ) -> InterpolateResult<ConfidenceExtrapolationResult<T>> {
        if let Some(config) = &self.confidence_config {
            let base_result = self.base_extrapolator.extrapolate(x)?;

            // Estimate uncertainty based on distance from domain boundaries
            let lower_bound = self.base_extrapolator.lower_bound();
            let upper_bound = self.base_extrapolator.upper_bound();

            // Calculate distance from nearest boundary
            let distance_from_domain = if x < lower_bound {
                lower_bound - x
            } else if x > upper_bound {
                x - upper_bound
            } else {
                T::zero() // Inside domain
            };

            // Uncertainty increases with distance from domain
            // Standard error grows linearly with distance (simple model)
            let base_uncertainty = T::from(0.01).unwrap_or_default(); // 1% base uncertainty
            let distance_factor = T::from(0.1).unwrap_or_default(); // 10% per unit distance
            let _standard_error = base_uncertainty + distance_factor * distance_from_domain;

            // Calculate confidence bounds based on confidence level
            // Using normal approximation: bounds = estimate ± z * standard_error
            let z_score = if config.confidence_level >= T::from(0.99).unwrap_or_default() {
                T::from(2.576).unwrap_or_default() // 99%
            } else if config.confidence_level >= T::from(0.95).unwrap_or_default() {
                T::from(1.96).unwrap_or_default() // 95%
            } else if config.confidence_level >= T::from(0.90).unwrap_or_default() {
                T::from(1.645).unwrap_or_default() // 90%
            } else {
                T::from(1.0).unwrap_or_default() // Default 1-sigma
            };

            let margin_of_error = z_score * _standard_error;
            let lower_bound_confidence = base_result - margin_of_error;
            let upper_bound_confidence = base_result + margin_of_error;

            Ok(ConfidenceExtrapolationResult {
                value: base_result,
                lower_bound: lower_bound_confidence,
                upper_bound: upper_bound_confidence,
                confidence_level: config.confidence_level,
            })
        } else {
            Err(InterpolateError::ComputationError(
                "Confidence extrapolation not configured".to_string(),
            ))
        }
    }

    /// Perform ensemble extrapolation
    pub fn extrapolate_ensemble(&self, x: T) -> InterpolateResult<T> {
        if let Some(config) = &self.ensemble_config {
            let mut results = Vec::new();
            let mut weights = Vec::new();

            // Collect results from all methods
            for (i, &method) in config.methods.iter().enumerate() {
                // Create a temporary extrapolator with this method
                let mut temp_extrapolator = self.base_extrapolator.clone();

                // Update the extrapolation method based on direction
                if x < temp_extrapolator.lower_bound() {
                    temp_extrapolator.lower_method = method;
                } else if x > temp_extrapolator.upper_bound() {
                    temp_extrapolator.upper_method = method;
                }

                if let Ok(result) = temp_extrapolator.extrapolate(x) {
                    results.push(result);
                    let weight = if let Some(w) = config.weights.as_ref() {
                        w.get(i).copied().unwrap_or(T::one())
                    } else {
                        T::one()
                    };
                    weights.push(weight);
                }
            }

            if results.is_empty() {
                return Err(InterpolateError::ComputationError(
                    "No ensemble methods produced valid results".to_string(),
                ));
            }

            // Combine results based on strategy
            match config.combination_strategy {
                EnsembleCombinationStrategy::Mean => {
                    let sum: T = results.iter().copied().fold(T::zero(), |acc, x| acc + x);
                    Ok(sum / T::from(results.len()).unwrap())
                }
                EnsembleCombinationStrategy::WeightedMean => {
                    let weighted_sum: T = results
                        .iter()
                        .zip(weights.iter())
                        .map(|(r, w)| *r * *w)
                        .fold(T::zero(), |acc, x| acc + x);
                    let weight_sum: T = weights.iter().copied().fold(T::zero(), |acc, x| acc + x);

                    if weight_sum.is_zero() {
                        return Err(InterpolateError::ComputationError(
                            "Zero total weight in ensemble".to_string(),
                        ));
                    }

                    Ok(weighted_sum / weight_sum)
                }
                EnsembleCombinationStrategy::Median => {
                    let mut sorted_results = results;
                    sorted_results.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let mid = sorted_results.len() / 2;

                    if sorted_results.len() % 2 == 0 {
                        let two = T::from(2.0).unwrap();
                        Ok((sorted_results[mid - 1] + sorted_results[mid]) / two)
                    } else {
                        Ok(sorted_results[mid])
                    }
                }
                EnsembleCombinationStrategy::BestMethod => {
                    // Use the method with highest confidence (simplified to first result)
                    Ok(results[0])
                }
                EnsembleCombinationStrategy::MinimumVariance => {
                    // Simplified implementation using equal weights
                    let sum: T = results.iter().copied().fold(T::zero(), |acc, x| acc + x);
                    Ok(sum / T::from(results.len()).unwrap())
                }
                EnsembleCombinationStrategy::BayesianAveraging => {
                    // Simplified implementation using uniform priors
                    let sum: T = results.iter().copied().fold(T::zero(), |acc, x| acc + x);
                    Ok(sum / T::from(results.len()).unwrap())
                }
                EnsembleCombinationStrategy::Voting => {
                    // For regression, use median as "majority vote"
                    let mut sorted_results = results;
                    sorted_results.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let mid = sorted_results.len() / 2;
                    Ok(sorted_results[mid])
                }
                EnsembleCombinationStrategy::Stacking => {
                    // Simplified stacking using equal weights
                    let sum: T = results.iter().copied().fold(T::zero(), |acc, x| acc + x);
                    Ok(sum / T::from(results.len()).unwrap())
                }
            }
        } else {
            Err(InterpolateError::ComputationError(
                "Ensemble extrapolation not configured".to_string(),
            ))
        }
    }

    /// Perform adaptive extrapolation
    pub fn extrapolate_adaptive(&self, x: T) -> InterpolateResult<T> {
        if let Some(_config) = &self.adaptive_config {
            // Simplified adaptive extrapolation
            // In a full implementation, this would analyze local data characteristics
            // and select the best method based on the selection criterion

            let candidate_methods = vec![
                ExtrapolationMethod::Linear,
                ExtrapolationMethod::Quadratic,
                ExtrapolationMethod::Cubic,
                ExtrapolationMethod::Exponential,
            ];

            let mut best_result = None;
            let mut _best_score = T::infinity();

            // Try each candidate method and select the best one
            for &method in &candidate_methods {
                let mut temp_extrapolator = self.base_extrapolator.clone();

                // Update the extrapolation method based on direction
                if x < temp_extrapolator.lower_bound() {
                    temp_extrapolator.lower_method = method;
                } else if x > temp_extrapolator.upper_bound() {
                    temp_extrapolator.upper_method = method;
                }

                if let Ok(result) = temp_extrapolator.extrapolate(x) {
                    if best_result.is_none() {
                        best_result = Some(result);
                        // In a full implementation, we'd compute a quality score here
                    }
                }
            }

            best_result.ok_or_else(|| {
                InterpolateError::ComputationError(
                    "No adaptive methods produced valid results".to_string(),
                )
            })
        } else {
            Err(InterpolateError::ComputationError(
                "Adaptive extrapolation not configured".to_string(),
            ))
        }
    }

    /// Perform autoregressive extrapolation
    pub fn extrapolate_autoregressive(&self, x: T) -> InterpolateResult<T> {
        if let Some(config) = &self.autoregressive_config {
            if let Some((x_data, y_data)) = &self.historical_data {
                // Fit AR model and predict
                let ar_coeffs = self.fit_ar_model(x_data, y_data, config.ar_order)?;
                self.ar_predict(&ar_coeffs, x_data, y_data, x, config)
            } else {
                Err(InterpolateError::ComputationError(
                    "Historical data required for autoregressive extrapolation".to_string(),
                ))
            }
        } else {
            Err(InterpolateError::ComputationError(
                "Autoregressive extrapolation not configured".to_string(),
            ))
        }
    }

    /// Fit autoregressive model to historical data
    fn fit_ar_model(
        &self,
        _x_data: &Array1<T>,
        y_data: &Array1<T>,
        order: usize,
    ) -> InterpolateResult<Array1<T>> {
        if y_data.len() < order + 1 {
            return Err(InterpolateError::ComputationError(
                "Insufficient data for AR model fitting".to_string(),
            ));
        }

        // Simple AR fitting using Yule-Walker equations (simplified version)
        let n = y_data.len();
        let mut coeffs = Array1::zeros(order);

        // For simplicity, use least squares approach
        // In practice, you'd use more sophisticated methods like Burg's method

        // Calculate autocorrelations
        let mut autocorr = Array1::zeros(order + 1);
        for lag in 0..=order {
            let mut sum = T::zero();
            let mut count = 0;

            for i in lag..n {
                sum += y_data[i] * y_data[i - lag];
                count += 1;
            }

            if count > 0 {
                autocorr[lag] = sum / T::from(count).unwrap_or(T::one());
            }
        }

        // Solve Yule-Walker equations (simplified)
        // For a proper implementation, you'd solve the full Toeplitz system
        for i in 0..order {
            if autocorr[0] != T::zero() {
                coeffs[i] = autocorr[i + 1] / autocorr[0];
            }
        }

        Ok(coeffs)
    }

    /// Make AR prediction
    fn ar_predict(
        &self,
        coeffs: &Array1<T>,
        x_data: &Array1<T>,
        y_data: &Array1<T>,
        x: T,
        _config: &AutoregressiveExtrapolationConfig<T>,
    ) -> InterpolateResult<T> {
        let order = coeffs.len();

        if y_data.len() < order {
            return Err(InterpolateError::ComputationError(
                "Insufficient data for AR prediction".to_string(),
            ));
        }

        // Use the last 'order' values to predict
        let mut prediction = T::zero();
        let start_idx = y_data.len() - order;

        for i in 0..order {
            prediction += coeffs[i] * y_data[start_idx + i];
        }

        // Adjust prediction based on distance from domain
        // This is a simplified approach - in practice you'd interpolate the time series
        let last_x = x_data[x_data.len() - 1];
        let extrapolation_distance = x - last_x;

        // Apply simple trend adjustment (very basic)
        if extrapolation_distance != T::zero() && y_data.len() >= 2 {
            let trend = (y_data[y_data.len() - 1] - y_data[y_data.len() - 2])
                / (x_data[x_data.len() - 1] - x_data[x_data.len() - 2]);
            prediction += trend * extrapolation_distance;
        }

        Ok(prediction)
    }

    /// Get access to the base extrapolator
    pub fn base(&self) -> &Extrapolator<T> {
        &self.base_extrapolator
    }

    /// Get mutable access to the base extrapolator
    pub fn base_mut(&mut self) -> &mut Extrapolator<T> {
        &mut self.base_extrapolator
    }

    /// Check if confidence estimation is enabled
    pub fn has_confidence(&self) -> bool {
        self.confidence_config.is_some()
    }

    /// Check if ensemble methods are enabled
    pub fn has_ensemble(&self) -> bool {
        self.ensemble_config.is_some()
    }

    /// Check if adaptive selection is enabled
    pub fn has_adaptive(&self) -> bool {
        self.adaptive_config.is_some()
    }

    /// Check if autoregressive modeling is enabled
    pub fn has_autoregressive(&self) -> bool {
        self.autoregressive_config.is_some()
    }

    /// Check if historical data is available
    pub fn has_historical_data(&self) -> bool {
        self.historical_data.is_some()
    }

    /// Get the number of available AR coefficients
    pub fn ar_model_order(&self) -> Option<usize> {
        self.autoregressive_config.as_ref().map(|c| c.ar_order)
    }

    /// Perform multiple extrapolations efficiently
    pub fn extrapolate_batch(&self, x_values: &[T]) -> Vec<InterpolateResult<T>> {
        x_values
            .iter()
            .map(|&x| self.extrapolate_advanced(x))
            .collect()
    }

    /// Get extrapolation method recommendations based on data characteristics
    pub fn recommend_methods(&self, x: T) -> Vec<ExtrapolationMethod> {
        let mut recommendations = Vec::new();

        // Basic recommendations based on position relative to domain
        let distance_from_lower = if x < self.base_extrapolator.lower_bound() {
            self.base_extrapolator.lower_bound() - x
        } else {
            T::zero()
        };

        let distance_from_upper = if x > self.base_extrapolator.upper_bound() {
            x - self.base_extrapolator.upper_bound()
        } else {
            T::zero()
        };

        let domain_width = self.base_extrapolator.domain_width();

        // For small extrapolation distances, recommend higher-order methods
        if distance_from_lower < domain_width * T::from(0.1).unwrap_or(T::one())
            || distance_from_upper < domain_width * T::from(0.1).unwrap_or(T::one())
        {
            recommendations.push(ExtrapolationMethod::Cubic);
            recommendations.push(ExtrapolationMethod::Quadratic);
        }

        // For moderate distances, recommend robust methods
        recommendations.push(ExtrapolationMethod::Linear);

        // For large distances, recommend asymptotic methods
        if distance_from_lower > domain_width * T::from(0.5).unwrap_or(T::one())
            || distance_from_upper > domain_width * T::from(0.5).unwrap_or(T::one())
        {
            recommendations.push(ExtrapolationMethod::Exponential);
            recommendations.push(ExtrapolationMethod::PowerLaw);
        }

        recommendations
    }
}
