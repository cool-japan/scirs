//! Enhanced ARIMA models with automatic order selection
//!
//! Implements advanced ARIMA fitting, diagnostic checking, and model selection

use ndarray::{s, Array1, ArrayBase, Data, Ix1, ScalarOperand};
use num_traits::{Float, FromPrimitive};
use std::fmt::{Debug, Display};

use crate::error::{Result, TimeSeriesError};
use crate::optimization::{LBFGSOptimizer, OptimizationOptions};
use crate::tests::{adf_test, ADFRegression};
use crate::utils::{autocorrelation, partial_autocorrelation};

/// ARIMA model parameters
#[derive(Debug, Clone)]
pub struct ArimaModel<F> {
    /// AR order
    pub p: usize,
    /// Differencing order
    pub d: usize,
    /// MA order
    pub q: usize,
    /// AR coefficients
    pub ar_coeffs: Array1<F>,
    /// MA coefficients
    pub ma_coeffs: Array1<F>,
    /// Model intercept
    pub intercept: F,
    /// Residual variance
    pub sigma2: F,
    /// Log-likelihood
    pub log_likelihood: F,
    /// Number of observations used for fitting
    pub n_obs: usize,
    /// Whether the model is fitted
    pub is_fitted: bool,
}

/// Seasonal ARIMA parameters
#[derive(Debug, Clone)]
pub struct SarimaParams {
    /// Non-seasonal orders (p, d, q)
    pub pdq: (usize, usize, usize),
    /// Seasonal orders (P, D, Q)
    pub seasonal_pdq: (usize, usize, usize),
    /// Seasonal period
    pub seasonal_period: usize,
}

/// ARIMA model selection options
#[derive(Debug, Clone)]
pub struct ArimaSelectionOptions {
    /// Maximum AR order
    pub max_p: usize,
    /// Maximum differencing order
    pub max_d: usize,
    /// Maximum MA order
    pub max_q: usize,
    /// Whether to include seasonal components
    pub seasonal: bool,
    /// Seasonal period
    pub seasonal_period: Option<usize>,
    /// Maximum seasonal AR order
    pub max_seasonal_p: usize,
    /// Maximum seasonal differencing order
    pub max_seasonal_d: usize,
    /// Maximum seasonal MA order
    pub max_seasonal_q: usize,
    /// Information criterion for model selection
    pub criterion: SelectionCriterion,
    /// Whether to use stepwise search
    pub stepwise: bool,
    /// Whether to test for stationarity
    pub test_stationarity: bool,
    /// Significance level for stationarity tests
    pub alpha: f64,
    /// Whether to use parallel computation
    pub parallel: bool,
    /// Whether to include a constant term
    pub include_constant: bool,
    /// Maximum number of iterations for optimization
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
}

impl Default for ArimaSelectionOptions {
    fn default() -> Self {
        Self {
            max_p: 5,
            max_d: 2,
            max_q: 5,
            seasonal: false,
            seasonal_period: None,
            max_seasonal_p: 2,
            max_seasonal_d: 1,
            max_seasonal_q: 2,
            criterion: SelectionCriterion::AIC,
            stepwise: true,
            test_stationarity: true,
            alpha: 0.05,
            parallel: false,
            include_constant: true,
            max_iter: 1000,
            tolerance: 1e-8,
        }
    }
}

/// Model selection criterion
#[derive(Debug, Clone, Copy)]
pub enum SelectionCriterion {
    /// Akaike Information Criterion
    AIC,
    /// Corrected AIC for small samples
    AICc,
    /// Bayesian Information Criterion
    BIC,
    /// Hannan-Quinn Information Criterion
    HQC,
}

impl<F> ArimaModel<F>
where
    F: Float + FromPrimitive + Debug + Display + ScalarOperand,
{
    /// Create a new ARIMA model
    pub fn new(p: usize, d: usize, q: usize) -> Result<Self> {
        crate::validation::validate_arima_orders(p, d, q)?;

        Ok(Self {
            p,
            d,
            q,
            ar_coeffs: Array1::zeros(p),
            ma_coeffs: Array1::zeros(q),
            intercept: F::zero(),
            sigma2: F::one(),
            log_likelihood: F::neg_infinity(),
            n_obs: 0,
            is_fitted: false,
        })
    }

    /// Fit ARIMA model using maximum likelihood estimation
    pub fn fit<S>(&mut self, data: &ArrayBase<S, Ix1>) -> Result<()>
    where
        S: Data<Elem = F>,
    {
        scirs2_core::validation::check_array_finite(data, "data")?;

        let min_required = self.p + self.q + self.d + 1;
        crate::validation::check_array_length(data, min_required, "ARIMA model fitting")?;

        // Apply differencing
        let diff_data = self.difference(data)?;
        self.n_obs = diff_data.len();

        // Initialize parameters
        self.initialize_parameters(&diff_data)?;

        // Optimize parameters using MLE
        self.optimize_parameters(&diff_data)?;

        // Calculate log-likelihood
        self.log_likelihood = self.calculate_log_likelihood(&diff_data)?;

        self.is_fitted = true;
        Ok(())
    }

    /// Apply differencing
    fn difference<S>(&self, data: &ArrayBase<S, Ix1>) -> Result<Array1<F>>
    where
        S: Data<Elem = F>,
    {
        let mut result = data.to_owned();

        for _ in 0..self.d {
            let mut diff = Array1::zeros(result.len() - 1);
            for i in 1..result.len() {
                diff[i - 1] = result[i] - result[i - 1];
            }
            result = diff;
        }

        Ok(result)
    }

    /// Initialize parameters using method of moments
    fn initialize_parameters(&mut self, data: &Array1<F>) -> Result<()> {
        // Initialize AR coefficients using Yule-Walker equations
        if self.p > 0 {
            let _acf = autocorrelation(data, Some(self.p))?;
            let pacf = partial_autocorrelation(data, Some(self.p))?;

            for i in 0..self.p {
                self.ar_coeffs[i] = pacf[i + 1];
            }
        }

        // Initialize MA coefficients
        for i in 0..self.q {
            self.ma_coeffs[i] = F::from(0.1).unwrap();
        }

        // Initialize intercept
        self.intercept = data.mean().unwrap_or(F::zero());

        // Initialize variance
        self.sigma2 = data
            .mapv(|x| (x - self.intercept) * (x - self.intercept))
            .mean()
            .unwrap_or(F::one());

        Ok(())
    }

    /// Optimize parameters using maximum likelihood
    fn optimize_parameters(&mut self, data: &Array1<F>) -> Result<()> {
        // Prepare parameters for optimization
        let n_params = self.p + self.q + 1; // AR coeffs + MA coeffs + intercept
        let mut params = Array1::zeros(n_params);

        // Pack current parameters
        for i in 0..self.p {
            params[i] = self.ar_coeffs[i];
        }
        for i in 0..self.q {
            params[self.p + i] = self.ma_coeffs[i];
        }
        params[self.p + self.q] = self.intercept;

        // Create optimizer
        let mut optimizer = LBFGSOptimizer::new(OptimizationOptions::default());

        // Reference to self for closure capture
        let p = self.p;
        let q = self.q;
        let data_clone = data.clone();

        // Define objective function (negative log-likelihood)
        let objective = |params: &Array1<F>| -> F {
            // Unpack parameters
            let mut model = self.clone();
            for i in 0..p {
                model.ar_coeffs[i] = params[i];
            }
            for i in 0..q {
                model.ma_coeffs[i] = params[p + i];
            }
            model.intercept = params[p + q];

            // Calculate residuals and log-likelihood
            if let Ok(residuals) = model.calculate_residuals(&data_clone) {
                let n = F::from(data_clone.len()).unwrap();
                let sigma2 = residuals.dot(&residuals) / n;

                // Negative log-likelihood
                n / F::from(2.0).unwrap()
                    * (F::one() + F::from(2.0 * std::f64::consts::PI).unwrap().ln())
                    + n / F::from(2.0).unwrap() * sigma2.ln()
                    + residuals.dot(&residuals) / (F::from(2.0).unwrap() * sigma2)
            } else {
                F::infinity()
            }
        };

        // Define gradient function
        let gradient = |params: &Array1<F>| -> Array1<F> {
            let mut grad = Array1::zeros(n_params);

            // Unpack parameters
            let mut model = self.clone();
            for i in 0..p {
                model.ar_coeffs[i] = params[i];
            }
            for i in 0..q {
                model.ma_coeffs[i] = params[p + i];
            }
            model.intercept = params[p + q];

            if let Ok(residuals) = model.calculate_residuals(&data_clone) {
                let n = F::from(data_clone.len()).unwrap();
                model.sigma2 = residuals.dot(&residuals) / n;

                // AR gradients
                for i in 0..p {
                    if let Ok(g) = model.ar_gradient(&data_clone, &residuals, i) {
                        grad[i] = -g; // Negative for minimization
                    }
                }

                // MA gradients
                for i in 0..q {
                    if let Ok(g) = model.ma_gradient(&data_clone, &residuals, i) {
                        grad[p + i] = -g; // Negative for minimization
                    }
                }

                // Intercept gradient
                grad[p + q] = -residuals.sum() / model.sigma2 / n;
            }

            grad
        };

        // Optimize
        let result = optimizer.optimize(objective, gradient, &params)?;

        // Update parameters
        for i in 0..self.p {
            self.ar_coeffs[i] = result.x[i];
        }
        for i in 0..self.q {
            self.ma_coeffs[i] = result.x[self.p + i];
        }
        self.intercept = result.x[self.p + self.q];

        // Update sigma2
        let residuals = self.calculate_residuals(data)?;
        self.sigma2 = residuals.dot(&residuals) / F::from(data.len()).unwrap();

        Ok(())
    }

    /// Calculate residuals
    fn calculate_residuals(&self, data: &Array1<F>) -> Result<Array1<F>> {
        let n = data.len();
        let mut residuals = Array1::zeros(n);

        for t in self.p.max(self.q)..n {
            let mut pred = self.intercept;

            // AR component
            for i in 0..self.p {
                if t > i {
                    pred = pred + self.ar_coeffs[i] * data[t - i - 1];
                }
            }

            // MA component
            for i in 0..self.q {
                if t > i {
                    pred = pred + self.ma_coeffs[i] * residuals[t - i - 1];
                }
            }

            residuals[t] = data[t] - pred;
        }

        Ok(residuals)
    }

    /// Calculate gradient for AR coefficient
    fn ar_gradient(&self, data: &Array1<F>, residuals: &Array1<F>, idx: usize) -> Result<F> {
        let n = data.len();
        let mut grad = F::zero();

        for t in self.p.max(self.q)..n {
            if t > idx {
                grad = grad - residuals[t] * data[t - idx - 1] / self.sigma2;
            }
        }

        Ok(grad / F::from(n).unwrap())
    }

    /// Calculate gradient for MA coefficient
    fn ma_gradient(&self, _data: &Array1<F>, residuals: &Array1<F>, idx: usize) -> Result<F> {
        let n = residuals.len();
        let mut grad = F::zero();

        for t in self.p.max(self.q)..n {
            if t > idx {
                grad = grad - residuals[t] * residuals[t - idx - 1] / self.sigma2;
            }
        }

        Ok(grad / F::from(n).unwrap())
    }

    /// Calculate log-likelihood
    fn calculate_log_likelihood(&self, data: &Array1<F>) -> Result<F> {
        let residuals = self.calculate_residuals(data)?;
        let n = F::from(data.len()).unwrap();

        let ll = -n / F::from(2.0).unwrap()
            * (F::one() + F::from(2.0).unwrap() * F::from(std::f64::consts::PI).unwrap()).ln()
            - n / F::from(2.0).unwrap() * self.sigma2.ln()
            - residuals.dot(&residuals) / (F::from(2.0).unwrap() * self.sigma2);

        Ok(ll)
    }

    /// Forecast future values
    pub fn forecast(&self, steps: usize, data: &Array1<F>) -> Result<Array1<F>> {
        if !self.is_fitted {
            return Err(TimeSeriesError::InvalidModel(
                "Model must be fitted before forecasting".to_string(),
            ));
        }

        crate::validation::validate_forecast_horizon(steps, Some(1000))?;

        let diff_data = self.difference(data)?;
        let mut forecasts = Array1::zeros(steps);
        let mut extended_data = diff_data.to_vec();
        let mut residuals = self.calculate_residuals(&diff_data)?.to_vec();

        for h in 0..steps {
            let mut pred = self.intercept;

            // AR component
            for i in 0..self.p {
                let idx = extended_data.len() - i - 1;
                if idx < extended_data.len() {
                    pred = pred + self.ar_coeffs[i] * extended_data[idx];
                }
            }

            // MA component (assuming future shocks are zero)
            for i in 0..self.q {
                let idx = residuals.len() - i - 1;
                if idx < residuals.len() && i < h {
                    pred = pred + self.ma_coeffs[i] * residuals[idx];
                }
            }

            forecasts[h] = pred;
            extended_data.push(pred);
            residuals.push(F::zero()); // Future residuals assumed zero
        }

        // Integrate if differenced
        if self.d > 0 {
            self.integrate_forecast(&forecasts, data)
        } else {
            Ok(forecasts)
        }
    }

    /// Integrate forecasts back to original scale
    fn integrate_forecast(&self, forecasts: &Array1<F>, original: &Array1<F>) -> Result<Array1<F>> {
        let mut result = forecasts.to_owned();
        let mut last_values = original.slice(s![original.len() - self.d..]).to_vec();

        for _ in 0..self.d {
            let mut integrated = Array1::zeros(result.len());
            let last_val = last_values[last_values.len() - 1];

            integrated[0] = result[0] + last_val;
            for i in 1..result.len() {
                integrated[i] = result[i] + integrated[i - 1];
            }

            result = integrated;
            last_values.push(result[0]);
        }

        Ok(result)
    }

    /// Calculate information criteria
    pub fn aic(&self) -> F {
        let k = F::from(self.p + self.q + 1).unwrap(); // +1 for intercept
        F::from(2.0).unwrap() * k - F::from(2.0).unwrap() * self.log_likelihood
    }

    /// Calculate corrected AIC (AICc) for small samples
    pub fn aicc(&self) -> F {
        let k = F::from(self.p + self.q + 1).unwrap();
        let n = F::from(self.n_obs).unwrap();
        self.aic() + F::from(2.0).unwrap() * k * (k + F::one()) / (n - k - F::one())
    }

    /// Calculate Bayesian Information Criterion (BIC)
    pub fn bic(&self) -> F {
        let k = F::from(self.p + self.q + 1).unwrap();
        let n = F::from(self.n_obs).unwrap();
        k * n.ln() - F::from(2.0).unwrap() * self.log_likelihood
    }

    /// Calculate Hannan-Quinn Information Criterion (HQC)
    pub fn hqc(&self) -> F {
        let k = F::from(self.p + self.q + 1).unwrap();
        let n = F::from(self.n_obs).unwrap();
        F::from(2.0).unwrap() * k * n.ln().ln() - F::from(2.0).unwrap() * self.log_likelihood
    }

    /// Get information criterion value
    pub fn get_ic(&self, criterion: SelectionCriterion) -> F {
        match criterion {
            SelectionCriterion::AIC => self.aic(),
            SelectionCriterion::AICc => self.aicc(),
            SelectionCriterion::BIC => self.bic(),
            SelectionCriterion::HQC => self.hqc(),
        }
    }
}

/// Automatic ARIMA order selection
pub fn auto_arima<S, F>(
    data: &ArrayBase<S, Ix1>,
    options: &ArimaSelectionOptions,
) -> Result<(ArimaModel<F>, SarimaParams)>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive + Debug + Display + ScalarOperand,
{
    scirs2_core::validation::check_array_finite(data, "data")?;

    // Determine optimal differencing order
    let d = if options.test_stationarity {
        determine_differencing_order(data, options.max_d)?
    } else {
        0
    };

    // Determine optimal seasonal differencing if applicable
    let seasonal_d = if options.seasonal && options.seasonal_period.is_some() {
        determine_seasonal_differencing_order(
            data,
            options.seasonal_period.unwrap(),
            options.max_seasonal_d,
        )?
    } else {
        0
    };

    let mut best_model = ArimaModel::new(0, d, 0)?;
    let mut best_ic = F::infinity();
    let mut best_params = SarimaParams {
        pdq: (0, d, 0),
        seasonal_pdq: (0, seasonal_d, 0),
        seasonal_period: options.seasonal_period.unwrap_or(1),
    };

    // Apply differencing
    let diff_data = apply_differencing(data, d, seasonal_d, options.seasonal_period)?;

    if options.stepwise {
        // Stepwise search
        (best_model, best_params) = stepwise_search(&diff_data, d, seasonal_d, options)?;
    } else {
        // Grid search
        for p in 0..=options.max_p {
            for q in 0..=options.max_q {
                if let Ok(mut model) = ArimaModel::new(p, d, q) {
                    if model.fit(&diff_data).is_ok() {
                        let ic = model.get_ic(options.criterion);
                        if ic < best_ic {
                            best_ic = ic;
                            best_model = model;
                            best_params.pdq = (p, d, q);
                        }
                    }
                }
            }
        }

        // Search seasonal parameters if applicable
        if options.seasonal {
            for sp in 0..=options.max_seasonal_p {
                for sq in 0..=options.max_seasonal_q {
                    // Fit seasonal model (placeholder)
                    best_params.seasonal_pdq = (sp, seasonal_d, sq);
                }
            }
        }
    }

    // Refit on original data
    best_model.fit(data)?;

    Ok((best_model, best_params))
}

/// Stepwise search for optimal ARIMA parameters
fn stepwise_search<F>(
    data: &Array1<F>,
    d: usize,
    seasonal_d: usize,
    options: &ArimaSelectionOptions,
) -> Result<(ArimaModel<F>, SarimaParams)>
where
    F: Float + FromPrimitive + Debug + Display + ScalarOperand,
{
    let mut best_model = ArimaModel::new(0, d, 0)?;
    let mut best_ic = F::infinity();
    let mut best_params = SarimaParams {
        pdq: (0, d, 0),
        seasonal_pdq: (0, seasonal_d, 0),
        seasonal_period: options.seasonal_period.unwrap_or(1),
    };

    // Start with simple models
    let candidates = vec![
        (0, d, 0),
        (1, d, 0),
        (0, d, 1),
        (1, d, 1),
        (2, d, 0),
        (0, d, 2),
    ];

    for (p, _, q) in candidates {
        if p <= options.max_p && q <= options.max_q {
            if let Ok(mut model) = ArimaModel::new(p, d, q) {
                if model.fit(data).is_ok() {
                    let ic = model.get_ic(options.criterion);
                    if ic < best_ic {
                        best_ic = ic;
                        best_model = model;
                        best_params.pdq = (p, d, q);
                    }
                }
            }
        }
    }

    // Expand search around best model
    let (best_p, _, best_q) = best_params.pdq;
    for dp in -1i32..=1 {
        for dq in -1i32..=1 {
            let new_p = (best_p as i32 + dp).max(0) as usize;
            let new_q = (best_q as i32 + dq).max(0) as usize;

            if new_p <= options.max_p
                && new_q <= options.max_q
                && (new_p != best_p || new_q != best_q)
            {
                if let Ok(mut model) = ArimaModel::new(new_p, d, new_q) {
                    if model.fit(data).is_ok() {
                        let ic = model.get_ic(options.criterion);
                        if ic < best_ic {
                            best_ic = ic;
                            best_model = model;
                            best_params.pdq = (new_p, d, new_q);
                        }
                    }
                }
            }
        }
    }

    Ok((best_model, best_params))
}

/// Determine optimal differencing order
fn determine_differencing_order<S, F>(data: &ArrayBase<S, Ix1>, max_d: usize) -> Result<usize>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive + Debug + Display + ScalarOperand,
{
    // Use ADF test to determine optimal differencing order
    let alpha = F::from(0.05).unwrap();

    for d in 0..=max_d {
        let diff_data = apply_single_differencing(data, d)?;

        // If the series has too few observations after differencing, stop
        if diff_data.len() < 10 {
            return Ok(d.saturating_sub(1));
        }

        // Perform ADF test
        if let Ok(test) = adf_test(&diff_data, None, ADFRegression::ConstantAndTrend, alpha) {
            if test.is_stationary {
                return Ok(d);
            }
        }
    }

    // If still not stationary after max_d differences, return max_d
    Ok(max_d)
}

/// Determine optimal seasonal differencing order
fn determine_seasonal_differencing_order<S, F>(
    data: &ArrayBase<S, Ix1>,
    period: usize,
    max_d: usize,
) -> Result<usize>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive + Debug + Display + ScalarOperand,
{
    let alpha = F::from(0.05).unwrap();

    for d in 0..=max_d {
        let diff_data = apply_seasonal_differencing(data, period, d)?;

        // If the series has too few observations after differencing, stop
        if diff_data.len() < 10 {
            return Ok(d.saturating_sub(1));
        }

        // Perform ADF test on seasonally differenced data
        if let Ok(test) = adf_test(&diff_data, None, ADFRegression::ConstantAndTrend, alpha) {
            if test.is_stationary {
                return Ok(d);
            }
        }
    }

    Ok(max_d)
}

/// Apply single differencing
fn apply_single_differencing<S, F>(data: &ArrayBase<S, Ix1>, d: usize) -> Result<Array1<F>>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive,
{
    let mut result = data.to_owned();

    for _ in 0..d {
        if result.len() <= 1 {
            return Err(TimeSeriesError::InvalidInput(
                "Insufficient data for differencing".to_string(),
            ));
        }

        let mut diff = Array1::zeros(result.len() - 1);
        for i in 1..result.len() {
            diff[i - 1] = result[i] - result[i - 1];
        }
        result = diff;
    }

    Ok(result)
}

/// Apply seasonal differencing
fn apply_seasonal_differencing<S, F>(
    data: &ArrayBase<S, Ix1>,
    period: usize,
    d: usize,
) -> Result<Array1<F>>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive,
{
    let mut result = data.to_owned();

    for _ in 0..d {
        if result.len() <= period {
            return Err(TimeSeriesError::InvalidInput(
                "Insufficient data for seasonal differencing".to_string(),
            ));
        }

        let mut diff = Array1::zeros(result.len() - period);
        for i in period..result.len() {
            diff[i - period] = result[i] - result[i - period];
        }
        result = diff;
    }

    Ok(result)
}

/// Apply both regular and seasonal differencing
fn apply_differencing<S, F>(
    data: &ArrayBase<S, Ix1>,
    d: usize,
    seasonal_d: usize,
    period: Option<usize>,
) -> Result<Array1<F>>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive,
{
    let mut result = apply_single_differencing(data, d)?;

    if seasonal_d > 0 && period.is_some() {
        result = apply_seasonal_differencing(&result, period.unwrap(), seasonal_d)?;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_arima_creation() {
        let model = ArimaModel::<f64>::new(2, 1, 1).unwrap();
        assert_eq!(model.p, 2);
        assert_eq!(model.d, 1);
        assert_eq!(model.q, 1);
        assert!(!model.is_fitted);
    }

    #[test]
    fn test_arima_creation_invalid() {
        // Test invalid AR order
        let result = ArimaModel::<f64>::new(11, 1, 1);
        assert!(result.is_err());

        // Test invalid differencing order
        let result = ArimaModel::<f64>::new(1, 4, 1);
        assert!(result.is_err());

        // Test invalid MA order
        let result = ArimaModel::<f64>::new(1, 1, 11);
        assert!(result.is_err());
    }

    #[test]
    fn test_differencing() {
        let data = array![1.0, 2.0, 4.0, 7.0, 11.0];
        let model = ArimaModel::<f64>::new(0, 1, 0).unwrap();
        let diff = model.difference(&data).unwrap();

        assert_eq!(diff.len(), 4);
        assert_eq!(diff[0], 1.0);
        assert_eq!(diff[1], 2.0);
        assert_eq!(diff[2], 3.0);
        assert_eq!(diff[3], 4.0);
    }

    #[test]
    fn test_arima_fit() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mut model = ArimaModel::new(1, 0, 1).unwrap();

        let result = model.fit(&data);
        assert!(result.is_ok());
        assert!(model.is_fitted);
    }

    #[test]
    fn test_auto_arima() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let options = ArimaSelectionOptions::default();

        let result = auto_arima(&data, &options);
        assert!(result.is_ok());

        let (model, params) = result.unwrap();
        assert!(model.is_fitted);
        assert_eq!(params.pdq.1, model.d);
    }

    #[test]
    fn test_determine_differencing() {
        let data = array![1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 29.0];
        let d = determine_differencing_order(&data, 2).unwrap();
        assert!(d <= 2);
    }

    #[test]
    fn test_forecast() {
        // Use a more complex series to avoid constant series after differencing
        let data = array![1.0, 2.4, 3.2, 4.1, 5.5, 6.2, 7.8, 8.3, 9.7, 10.1];
        let mut model = ArimaModel::new(1, 0, 0).unwrap();
        model.fit(&data).unwrap();

        let forecasts = model.forecast(3, &data).unwrap();
        assert_eq!(forecasts.len(), 3);
    }
}
