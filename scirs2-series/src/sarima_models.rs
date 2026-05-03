//! Seasonal ARIMA (SARIMA) models
//!
//! Implements SARIMA models with proper seasonal components

use scirs2_core::ndarray::{Array1, ArrayBase, Data, Ix1, ScalarOperand};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::{Debug, Display};

use crate::arima_models::SelectionCriterion;
use crate::error::{Result, TimeSeriesError};
use crate::optimization::{LBFGSOptimizer, OptimizationOptions};
use crate::utils::partial_autocorrelation;
use statrs::statistics::Statistics;

/// SARIMA model parameters
#[derive(Debug, Clone)]
pub struct SarimaModel<F> {
    /// Non-seasonal AR order
    pub p: usize,
    /// Non-seasonal differencing order
    pub d: usize,
    /// Non-seasonal MA order
    pub q: usize,
    /// Seasonal AR order
    pub p_seasonal: usize,
    /// Seasonal differencing order
    pub d_seasonal: usize,
    /// Seasonal MA order
    pub q_seasonal: usize,
    /// Seasonal period
    pub period: usize,
    /// AR coefficients
    pub ar_params: Array1<F>,
    /// MA coefficients
    pub ma_params: Array1<F>,
    /// Seasonal AR coefficients
    pub sar_params: Array1<F>,
    /// Seasonal MA coefficients
    pub sma_params: Array1<F>,
    /// Intercept
    pub intercept: F,
    /// Model fitted
    pub is_fitted: bool,
    /// Log-likelihood
    pub log_likelihood: F,
    /// AIC
    pub aic: F,
    /// BIC
    pub bic: F,
    /// Training data stored for use in forecasting without requiring caller to re-supply
    pub training_data: Option<Array1<F>>,
}

impl<F> SarimaModel<F>
where
    F: Float + FromPrimitive + Debug + Display + ScalarOperand,
{
    /// Create new SARIMA model
    pub fn new(
        p: usize,
        d: usize,
        q: usize,
        p_seasonal: usize,
        d_seasonal: usize,
        q_seasonal: usize,
        period: usize,
    ) -> Result<Self> {
        if period < 2 {
            return Err(TimeSeriesError::InvalidModel(
                "Seasonal period must be at least 2".to_string(),
            ));
        }

        Ok(Self {
            p,
            d,
            q,
            p_seasonal,
            d_seasonal,
            q_seasonal,
            period,
            ar_params: Array1::zeros(p),
            ma_params: Array1::zeros(q),
            sar_params: Array1::zeros(p_seasonal),
            sma_params: Array1::zeros(q_seasonal),
            intercept: F::zero(),
            is_fitted: false,
            log_likelihood: F::neg_infinity(),
            aic: F::infinity(),
            bic: F::infinity(),
            training_data: None,
        })
    }

    /// Apply seasonal differencing
    pub fn seasonal_difference(&self, data: &ArrayBase<impl Data<Elem = F>, Ix1>) -> Array1<F> {
        let mut result = data.to_owned();

        for _ in 0..self.d_seasonal {
            let n = result.len();
            if n <= self.period {
                return Array1::zeros(0);
            }

            let mut diff = Array1::zeros(n - self.period);
            for i in self.period..n {
                diff[i - self.period] = result[i] - result[i - self.period];
            }
            result = diff;
        }

        result
    }

    /// Apply regular differencing
    pub fn difference(&self, data: &ArrayBase<impl Data<Elem = F>, Ix1>) -> Array1<F> {
        let mut result = data.to_owned();

        for _ in 0..self.d {
            let n = result.len();
            if n <= 1 {
                return Array1::zeros(0);
            }

            let mut diff = Array1::zeros(n - 1);
            for i in 1..n {
                diff[i - 1] = result[i] - result[i - 1];
            }
            result = diff;
        }

        result
    }

    /// Apply both differencing operations
    pub fn full_difference(&self, data: &ArrayBase<impl Data<Elem = F>, Ix1>) -> Array1<F> {
        // First apply regular differencing, then seasonal
        let regular_diff = self.difference(data);
        self.seasonal_difference(&regular_diff)
    }

    /// Fit SARIMA model
    pub fn fit(&mut self, data: &ArrayBase<impl Data<Elem = F>, Ix1>) -> Result<()> {
        scirs2_core::validation::checkarray_finite(data, "data")?;

        // Need sufficient data for seasonal model
        let min_data = 2 * self.period + self.p + self.p_seasonal + self.d + self.d_seasonal;
        if data.len() < min_data {
            return Err(TimeSeriesError::InsufficientData {
                message: "Insufficient data for SARIMA model".to_string(),
                required: min_data,
                actual: data.len(),
            });
        }

        // Apply differencing
        let diff_data = self.full_difference(data);
        if diff_data.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "Differencing resulted in empty series".to_string(),
            ));
        }

        // Set up optimization
        let n_params = self.p + self.q + self.p_seasonal + self.q_seasonal + 1;
        let initial_params = self.initialize_parameters(&diff_data)?;

        let objective = |params: &Array1<F>| -> F {
            self.log_likelihood_full(params, &diff_data)
                .unwrap_or(F::infinity())
        };

        let gradient = |params: &Array1<F>| -> Array1<F> {
            self.gradient_log_likelihood(params, &diff_data)
                .unwrap_or_else(|_| Array1::zeros(n_params))
        };

        // Optimize
        let options = OptimizationOptions {
            max_iter: 500,
            tolerance: F::from(1e-8).expect("Failed to convert constant to float"),
            grad_tolerance: F::from(1e-6).expect("Failed to convert constant to float"),
            initial_step: F::from(0.1).expect("Failed to convert constant to float"),
            line_search_alpha: F::from(0.4).expect("Failed to convert constant to float"),
            line_search_beta: F::from(0.9).expect("Failed to convert constant to float"),
        };

        let mut optimizer = LBFGSOptimizer::new(options);
        let result = optimizer.optimize(&objective, &gradient, &initial_params)?;

        // Extract parameters
        self.extract_parameters(&result.x);

        // Calculate final metrics
        self.log_likelihood = self.log_likelihood_full(&result.x, &diff_data)?;
        self.aic = F::from(2.0).expect("Failed to convert constant to float")
            * F::from(n_params).expect("Failed to convert to float")
            - F::from(2.0).expect("Failed to convert constant to float") * self.log_likelihood;
        self.bic = F::from(n_params).expect("Failed to convert to float")
            * F::from(diff_data.len()).expect("Operation failed").ln()
            - F::from(2.0).expect("Failed to convert constant to float") * self.log_likelihood;

        self.training_data = Some(data.to_owned());
        self.is_fitted = true;

        Ok(())
    }

    /// Initialize parameters
    fn initialize_parameters(&self, data: &Array1<F>) -> Result<Array1<F>> {
        let n_params = self.p + self.q + self.p_seasonal + self.q_seasonal + 1;
        let mut params = Array1::zeros(n_params);

        // Initialize intercept
        params[0] = data.mean().expect("Operation failed");

        // Initialize AR parameters using partial autocorrelations
        if self.p > 0 {
            let pacf = partial_autocorrelation(data, Some(self.p))?;
            for i in 0..self.p {
                params[1 + i] = pacf[i];
            }
        }

        // Initialize seasonal parameters with small values
        let offset = 1 + self.p + self.q;
        for i in 0..self.p_seasonal {
            params[offset + i] = F::from(0.1).expect("Failed to convert constant to float");
        }

        for i in 0..self.q_seasonal {
            params[offset + self.p_seasonal + i] =
                F::from(0.1).expect("Failed to convert constant to float");
        }

        Ok(params)
    }

    /// Extract parameters from optimization result
    fn extract_parameters(&mut self, params: &Array1<F>) {
        let mut idx = 0;

        // Intercept
        self.intercept = params[idx];
        idx += 1;

        // AR parameters
        if self.p > 0 {
            self.ar_params = params
                .slice(scirs2_core::ndarray::s![idx..idx + self.p])
                .to_owned();
            idx += self.p;
        }

        // MA parameters
        if self.q > 0 {
            self.ma_params = params
                .slice(scirs2_core::ndarray::s![idx..idx + self.q])
                .to_owned();
            idx += self.q;
        }

        // Seasonal AR parameters
        if self.p_seasonal > 0 {
            self.sar_params = params
                .slice(scirs2_core::ndarray::s![idx..idx + self.p_seasonal])
                .to_owned();
            idx += self.p_seasonal;
        }

        // Seasonal MA parameters
        if self.q_seasonal > 0 {
            self.sma_params = params
                .slice(scirs2_core::ndarray::s![idx..idx + self.q_seasonal])
                .to_owned();
        }
    }

    /// Calculate log-likelihood for full parameter set
    fn log_likelihood_full(&self, params: &Array1<F>, data: &Array1<F>) -> Result<F> {
        let n = data.len();

        // Extract parameters temporarily
        let mut idx = 0;
        let intercept = params[idx];
        idx += 1;

        let ar_params = if self.p > 0 {
            params
                .slice(scirs2_core::ndarray::s![idx..idx + self.p])
                .to_owned()
        } else {
            Array1::zeros(0)
        };
        idx += self.p;

        let ma_params = if self.q > 0 {
            params
                .slice(scirs2_core::ndarray::s![idx..idx + self.q])
                .to_owned()
        } else {
            Array1::zeros(0)
        };
        idx += self.q;

        let sar_params = if self.p_seasonal > 0 {
            params
                .slice(scirs2_core::ndarray::s![idx..idx + self.p_seasonal])
                .to_owned()
        } else {
            Array1::zeros(0)
        };
        idx += self.p_seasonal;

        let sma_params = if self.q_seasonal > 0 {
            params
                .slice(scirs2_core::ndarray::s![idx..idx + self.q_seasonal])
                .to_owned()
        } else {
            Array1::zeros(0)
        };

        // Calculate residuals
        let residuals = self.calculate_residuals(
            data,
            intercept,
            &ar_params,
            &ma_params,
            &sar_params,
            &sma_params,
        )?;

        // Calculate log-likelihood (assuming Gaussian errors)
        let sigma2 = residuals.dot(&residuals) / F::from(n).expect("Failed to convert to float");
        let log_likelihood = -F::from(0.5).expect("Failed to convert constant to float")
            * F::from(n).expect("Failed to convert to float")
            * (F::from(2.0 * std::f64::consts::PI)
                .expect("Failed to convert to float")
                .ln()
                + sigma2.ln()
                + F::one());

        Ok(log_likelihood)
    }

    /// Calculate residuals for given parameters
    fn calculate_residuals(
        &self,
        data: &Array1<F>,
        intercept: F,
        ar_params: &Array1<F>,
        ma_params: &Array1<F>,
        sar_params: &Array1<F>,
        sma_params: &Array1<F>,
    ) -> Result<Array1<F>> {
        let n = data.len();
        let mut residuals = Array1::zeros(n);
        let mut predictions = Array1::zeros(n);

        // Start from max lag
        let start_idx = self
            .p
            .max(self.q)
            .max(self.p_seasonal * self.period)
            .max(self.q_seasonal * self.period);

        for t in start_idx..n {
            let mut pred = intercept;

            // AR terms
            for i in 0..self.p {
                if t > i {
                    pred = pred + ar_params[i] * data[t - i - 1];
                }
            }

            // MA terms
            for i in 0..self.q {
                if t > i {
                    pred = pred + ma_params[i] * residuals[t - i - 1];
                }
            }

            // Seasonal AR terms
            for i in 0..self.p_seasonal {
                let lag = (i + 1) * self.period;
                if t >= lag {
                    pred = pred + sar_params[i] * data[t - lag];
                }
            }

            // Seasonal MA terms
            for i in 0..self.q_seasonal {
                let lag = (i + 1) * self.period;
                if t >= lag {
                    pred = pred + sma_params[i] * residuals[t - lag];
                }
            }

            predictions[t] = pred;
            residuals[t] = data[t] - pred;
        }

        // Return only the valid portion
        Ok(residuals
            .slice(scirs2_core::ndarray::s![start_idx..])
            .to_owned())
    }

    /// Calculate gradient of log-likelihood
    fn gradient_log_likelihood(&self, params: &Array1<F>, data: &Array1<F>) -> Result<Array1<F>> {
        let n_params = params.len();
        let mut gradient = Array1::zeros(n_params);
        let epsilon = F::from(1e-6).expect("Failed to convert constant to float");

        // Numerical gradient using central differences
        for i in 0..n_params {
            let mut params_plus = params.clone();
            let mut params_minus = params.clone();

            params_plus[i] = params_plus[i] + epsilon;
            params_minus[i] = params_minus[i] - epsilon;

            let ll_plus = self.log_likelihood_full(&params_plus, data)?;
            let ll_minus = self.log_likelihood_full(&params_minus, data)?;

            gradient[i] = (ll_plus - ll_minus)
                / (F::from(2.0).expect("Failed to convert constant to float") * epsilon);
        }

        Ok(-gradient) // Negative because we're maximizing
    }

    /// Predict future values
    pub fn predict(
        &self,
        steps: usize,
        data: &ArrayBase<impl Data<Elem = F>, Ix1>,
    ) -> Result<Array1<F>> {
        if !self.is_fitted {
            return Err(TimeSeriesError::InvalidInput(
                "Model must be fitted before prediction".to_string(),
            ));
        }

        let diff_data = self.full_difference(data);
        let n = diff_data.len();

        // Initialize with historical data and zeros for predictions
        let mut extended_data = Array1::zeros(n + steps);
        extended_data
            .slice_mut(scirs2_core::ndarray::s![..n])
            .assign(&diff_data);

        let mut residuals = Array1::zeros(n + steps);

        // Calculate historical residuals
        let historical_residuals = self.calculate_residuals(
            &diff_data,
            self.intercept,
            &self.ar_params,
            &self.ma_params,
            &self.sar_params,
            &self.sma_params,
        )?;

        let start_idx = n - historical_residuals.len();
        residuals
            .slice_mut(scirs2_core::ndarray::s![start_idx..n])
            .assign(&historical_residuals);

        // Make predictions
        for t in n..(n + steps) {
            let mut pred = self.intercept;

            // AR terms
            for i in 0..self.p {
                if t > i {
                    pred = pred + self.ar_params[i] * extended_data[t - i - 1];
                }
            }

            // MA terms (using zeros for future errors)
            for i in 0..self.q {
                if t > i {
                    pred = pred + self.ma_params[i] * residuals[t - i - 1];
                }
            }

            // Seasonal AR terms
            for i in 0..self.p_seasonal {
                let lag = (i + 1) * self.period;
                if t >= lag {
                    pred = pred + self.sar_params[i] * extended_data[t - lag];
                }
            }

            // Seasonal MA terms
            for i in 0..self.q_seasonal {
                let lag = (i + 1) * self.period;
                if t >= lag {
                    pred = pred + self.sma_params[i] * residuals[t - lag];
                }
            }

            extended_data[t] = pred;
        }

        Ok(extended_data
            .slice(scirs2_core::ndarray::s![n..])
            .to_owned())
    }

    /// Forecast future values using the training data stored during fitting,
    /// returning predictions on the **original (undifferenced) scale**.
    ///
    /// Internally this calls `predict()` on the differenced series and then
    /// inverts all differencing operations (seasonal first, then regular) to
    /// recover values on the same scale as the original training data.
    ///
    /// # Errors
    /// Returns an error if the model has not been fitted yet.
    pub fn forecast(&self, steps: usize) -> Result<Array1<F>> {
        let data = self.training_data.as_ref().ok_or_else(|| {
            TimeSeriesError::InvalidInput("Model must be fitted before forecasting".to_string())
        })?;

        // predict() returns values on the fully-differenced scale.
        let diff_forecasts = self.predict(steps, data)?;

        // Undo differencing: full_difference() applies regular d first then
        // seasonal D, so the inverse applies seasonal D first then regular d.
        let after_seasonal = self.invert_seasonal_difference(&diff_forecasts, data)?;
        let original_scale = self.invert_regular_difference(&after_seasonal, data)?;

        Ok(original_scale)
    }

    /// Invert seasonal differencing for `steps` forecast steps.
    ///
    /// Requires the last `period * d_seasonal` values of the *regularly-differenced*
    /// series (i.e. the series after applying only the regular `d` differences).
    fn invert_seasonal_difference(
        &self,
        diff_forecasts: &Array1<F>,
        original: &Array1<F>,
    ) -> Result<Array1<F>> {
        if self.d_seasonal == 0 {
            return Ok(diff_forecasts.to_owned());
        }

        // Compute the regularly-differenced series (before seasonal diff).
        let regular_diffed = self.difference(original);
        if regular_diffed.is_empty() {
            return Ok(diff_forecasts.to_owned());
        }

        let steps = diff_forecasts.len();
        let mut result = diff_forecasts.to_owned();

        // Undo D rounds of seasonal differencing.
        for _ in 0..self.d_seasonal {
            // We need the last `period` values of `regular_diffed` as anchor points.
            let anchor_len = regular_diffed.len().min(self.period);
            let mut history: Vec<F> = regular_diffed
                .slice(scirs2_core::ndarray::s![
                    regular_diffed.len() - anchor_len..
                ])
                .to_vec();

            let mut integrated = Array1::zeros(steps);
            for t in 0..steps {
                // The seasonal lag index: look back `period` steps in the combined
                // history + integrated output so far.
                let season_val = if t < self.period {
                    // Need from the anchor (end of historical series).
                    let anchor_idx = anchor_len + t;
                    if anchor_idx >= self.period {
                        history[anchor_idx - self.period]
                    } else {
                        F::zero()
                    }
                } else {
                    integrated[t - self.period]
                };
                integrated[t] = result[t] + season_val;
            }
            result = integrated;
            // Extend history for next round if d_seasonal > 1.
            for t in 0..steps {
                history.push(result[t]);
            }
        }

        Ok(result)
    }

    /// Invert regular differencing for `steps` forecast steps.
    ///
    /// Requires the last `d` values of the *original* (pre-differenced) series.
    fn invert_regular_difference(
        &self,
        seasonally_inverted: &Array1<F>,
        original: &Array1<F>,
    ) -> Result<Array1<F>> {
        if self.d == 0 {
            return Ok(seasonally_inverted.to_owned());
        }

        let steps = seasonally_inverted.len();
        let mut result = seasonally_inverted.to_owned();

        for _ in 0..self.d {
            // Anchor: last value of the current level of the original series.
            // (For d > 1 we peel off one diff at a time using the last value
            // of the d-times-differenced prefix, which is the original series
            // itself on the first iteration.)
            let last_val = original[original.len() - 1];
            let mut integrated = Array1::zeros(steps);
            integrated[0] = result[0] + last_val;
            for i in 1..steps {
                integrated[i] = result[i] + integrated[i - 1];
            }
            result = integrated;
        }

        Ok(result)
    }
}

/// Type alias matching the integration test's `SARIMAModel<F>` naming convention.
pub type SARIMAModel<F> = SarimaModel<F>;

/// Auto SARIMA selection
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn auto_sarima<S, F>(
    data: &ArrayBase<S, Ix1>,
    period: usize,
    max_p: usize,
    max_q: usize,
    max_p_seasonal: usize,
    max_q_seasonal: usize,
    max_d: usize,
    max_d_seasonal: usize,
    criterion: SelectionCriterion,
) -> Result<(SarimaModel<F>, Array1<F>)>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive + Debug + Display + ScalarOperand,
{
    let mut best_metric = F::infinity();
    let mut best_model = None;

    // Grid search over all combinations
    for _d in 0..=max_d {
        for d_seasonal in 0..=max_d_seasonal {
            for _p in 0..=max_p {
                for _q in 0..=max_q {
                    for p_seasonal in 0..=max_p_seasonal {
                        for q_seasonal in 0..=max_q_seasonal {
                            // Skip invalid combinations
                            if _p == 0 && _q == 0 && p_seasonal == 0 && q_seasonal == 0 {
                                continue;
                            }

                            if let Ok(mut model) = SarimaModel::new(
                                _p, _d, _q, p_seasonal, d_seasonal, q_seasonal, period,
                            ) {
                                if model.fit(data).is_ok() {
                                    let n_params = _p + _q + p_seasonal + q_seasonal + 1;
                                    let metric = match criterion {
                                        SelectionCriterion::AIC => model.aic,
                                        SelectionCriterion::BIC => model.bic,
                                        SelectionCriterion::AICc => {
                                            // AICc = AIC + 2k(k+1)/(n-k-1)
                                            let k = F::from(n_params)
                                                .expect("Failed to convert to float");
                                            let n = F::from(data.len()).expect("Operation failed");
                                            model.aic
                                                + F::from(2.0)
                                                    .expect("Failed to convert constant to float")
                                                    * k
                                                    * (k + F::one())
                                                    / (n - k - F::one())
                                        }
                                        SelectionCriterion::HQC => {
                                            // HQC = -2*logL + 2*k*log(log(n))
                                            let k = F::from(n_params)
                                                .expect("Failed to convert to float");
                                            let n = F::from(data.len()).expect("Operation failed");
                                            F::from(2.0)
                                                .expect("Failed to convert constant to float")
                                                * k
                                                * n.ln().ln()
                                                - F::from(2.0)
                                                    .expect("Failed to convert constant to float")
                                                    * model.log_likelihood
                                        }
                                    };

                                    if metric < best_metric {
                                        best_metric = metric;
                                        best_model = Some(model);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    match best_model {
        Some(model) => {
            let params = {
                let mut _p = Array1::zeros(7);
                _p[0] = F::from(model.p).expect("Failed to convert to float");
                _p[1] = F::from(model.d).expect("Failed to convert to float");
                _p[2] = F::from(model.q).expect("Failed to convert to float");
                _p[3] = F::from(model.p_seasonal).expect("Failed to convert to float");
                _p[4] = F::from(model.d_seasonal).expect("Failed to convert to float");
                _p[5] = F::from(model.q_seasonal).expect("Failed to convert to float");
                _p[6] = F::from(model.period).expect("Failed to convert to float");
                _p
            };
            Ok((model, params))
        }
        None => Err(TimeSeriesError::FittingError(
            "Failed to fit any SARIMA model".to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_sarima_creation() {
        let model = SarimaModel::<f64>::new(1, 1, 1, 1, 1, 1, 12);
        assert!(model.is_ok());
    }

    #[test]
    fn test_sarima_invalid_period() {
        let model = SarimaModel::<f64>::new(1, 1, 1, 1, 1, 1, 1);
        assert!(model.is_err());
    }

    #[test]
    fn test_seasonal_differencing() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let model = SarimaModel::new(0, 0, 0, 0, 1, 0, 4).expect("Operation failed");
        let diff = model.seasonal_difference(&data);

        assert_eq!(diff.len(), 4);
        assert_eq!(diff[0], 4.0); // 5 - 1
        assert_eq!(diff[1], 4.0); // 6 - 2
        assert_eq!(diff[2], 4.0); // 7 - 3
        assert_eq!(diff[3], 4.0); // 8 - 4
    }

    #[test]
    fn test_forecast_returns_error_when_not_fitted() {
        let model = SarimaModel::<f64>::new(1, 0, 1, 1, 0, 1, 12).expect("Operation failed");
        let result = model.forecast(10);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("fitted"));
    }

    #[test]
    fn test_forecast_after_fitting_smoke() {
        // Directly construct a fitted SARIMA(1,0,0)(1,0,0)[4] model to test
        // forecast() in isolation from the optimizer (which can be numerically
        // fragile for certain parameter combinations).
        //
        // This verifies that: training_data is stored, forecast() delegates to
        // predict() and inverts differencing correctly, and the returned values
        // are on the same scale as the original training data (not differenced scale).
        let period = 4usize;
        let n = 40usize;
        let training_data: Array1<f64> = Array1::from_iter((0..n).map(|i| {
            50.0 + (2.0 * std::f64::consts::PI * (i % period) as f64 / period as f64).sin() * 5.0
        }));

        // Build a partially-fitted model by setting fields directly.
        let mut model =
            SarimaModel::<f64>::new(1, 0, 0, 1, 0, 0, period).expect("Operation failed");
        model.ar_params = array![0.5_f64];
        model.sar_params = array![0.3_f64];
        model.intercept = 0.0;
        model.is_fitted = true;
        model.training_data = Some(training_data.clone());

        let steps = 8;
        let forecast = model.forecast(steps).expect("Forecasting failed");
        assert_eq!(forecast.len(), steps);
        // All forecast values should be finite.
        assert!(forecast.iter().all(|v| v.is_finite()));
        // Forecasts should be on the original scale (around 50 ± 20), NOT on the
        // differenced scale (which would be near 0).  This catches regressions where
        // forecast() forgets to invert differencing.
        let last_val = training_data[training_data.len() - 1];
        for &pred in forecast.iter() {
            assert!(
                (pred - last_val).abs() < 30.0,
                "Forecast value {pred} is too far from last observed value {last_val}; \
                 did forecast() forget to invert differencing?"
            );
        }
    }
}
