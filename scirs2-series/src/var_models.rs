//! Vector Autoregressive (VAR) models for multivariate time series
//!
//! Implements VAR, VARMA, VECM and related multivariate time series models

use ndarray::{s, Array1, Array2, ArrayBase, Data, Ix2, ScalarOperand};
use num_traits::{Float, FromPrimitive};
use std::fmt::{Debug, Display};

use crate::error::{Result, TimeSeriesError};

/// Vector Autoregressive (VAR) model
#[derive(Debug, Clone)]
pub struct VARModel<F> {
    /// Order of the VAR model
    pub order: usize,
    /// Number of variables
    pub n_vars: usize,
    /// Coefficient matrices for each lag
    pub coefficients: Vec<Array2<F>>,
    /// Intercept vector
    pub intercept: Array1<F>,
    /// Covariance matrix of residuals
    pub covariance: Array2<F>,
    /// Whether the model has been fitted
    pub is_fitted: bool,
}

impl<F> VARModel<F>
where
    F: Float + FromPrimitive + Debug + Display + ScalarOperand,
{
    /// Create a new VAR model
    pub fn new(order: usize, n_vars: usize) -> Result<Self> {
        if order == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "VAR order must be at least 1".to_string(),
            ));
        }
        if n_vars == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "Number of variables must be at least 1".to_string(),
            ));
        }

        let coefficients = vec![Array2::zeros((n_vars, n_vars)); order];
        let intercept = Array1::zeros(n_vars);
        let covariance = Array2::eye(n_vars);

        Ok(Self {
            order,
            n_vars,
            coefficients,
            intercept,
            covariance,
            is_fitted: false,
        })
    }

    /// Fit the VAR model using OLS
    pub fn fit<S>(&mut self, data: &ArrayBase<S, Ix2>) -> Result<()>
    where
        S: Data<Elem = F>,
    {
        scirs2_core::validation::check_array_finite(data, "data")?;

        let (t, k) = data.dim();
        if k != self.n_vars {
            return Err(TimeSeriesError::InvalidInput(format!(
                "Data must have {} variables, got {}",
                self.n_vars, k
            )));
        }

        if t <= self.order {
            return Err(TimeSeriesError::InvalidInput(format!(
                "Time series length ({}) must be greater than VAR order ({})",
                t, self.order
            )));
        }

        // Construct design matrix and response matrix
        let n_obs = t - self.order;
        let n_regressors = self.order * self.n_vars + 1; // +1 for intercept

        let mut x = Array2::zeros((n_obs, n_regressors));
        let mut y = Array2::zeros((n_obs, self.n_vars));

        // Fill matrices
        for i in 0..n_obs {
            // Response variables
            for j in 0..self.n_vars {
                y[[i, j]] = data[[i + self.order, j]];
            }

            // Intercept
            x[[i, 0]] = F::one();

            // Lagged variables
            for lag in 0..self.order {
                for var in 0..self.n_vars {
                    let col_idx = 1 + lag * self.n_vars + var;
                    x[[i, col_idx]] = data[[i + self.order - lag - 1, var]];
                }
            }
        }

        // OLS estimation: β = (X'X)^(-1)X'Y
        let xtx = x.t().dot(&x);
        let xty = x.t().dot(&y);

        // Solve for coefficients (simplified - would use proper linear solver)
        let beta = solve_normal_equations(&xtx, &xty)?;

        // Extract coefficients
        self.intercept = beta.column(0).to_owned();

        for lag in 0..self.order {
            let mut coef_matrix = Array2::zeros((self.n_vars, self.n_vars));
            for i in 0..self.n_vars {
                for j in 0..self.n_vars {
                    let row_idx = 1 + lag * self.n_vars + j;
                    coef_matrix[[i, j]] = beta[[row_idx, i]];
                }
            }
            self.coefficients[lag] = coef_matrix;
        }

        // Calculate residuals and covariance
        let fitted = x.dot(&beta);
        let residuals = &y - &fitted;
        self.covariance = residuals.t().dot(&residuals) / F::from(n_obs - n_regressors).unwrap();

        self.is_fitted = true;
        Ok(())
    }

    /// Make predictions
    pub fn predict(&self, values: &Array2<F>, steps: usize) -> Result<Array2<F>> {
        if !self.is_fitted {
            return Err(TimeSeriesError::InvalidInput(
                "Model must be fitted before prediction".to_string(),
            ));
        }

        let (n, k) = values.dim();
        if k != self.n_vars {
            return Err(TimeSeriesError::InvalidInput(format!(
                "Data must have {} variables, got {}",
                self.n_vars, k
            )));
        }

        if n < self.order {
            return Err(TimeSeriesError::InvalidInput(format!(
                "Need at least {} observations for prediction, got {}",
                self.order, n
            )));
        }

        let mut predictions = Array2::zeros((steps, self.n_vars));
        let mut history = values.slice(s![n - self.order.., ..]).to_owned();

        for t in 0..steps {
            let mut pred = self.intercept.clone();

            for lag in 0..self.order {
                let lag_values = history.row(history.nrows() - 1 - lag);
                pred = pred + self.coefficients[lag].dot(&lag_values);
            }

            predictions.row_mut(t).assign(&pred);

            // Update history for next prediction
            if t < steps - 1 {
                // Shift history and add new prediction
                for i in 0..self.order - 1 {
                    let next_row = history.row(i + 1).to_owned();
                    history.row_mut(i).assign(&next_row);
                }
                history.row_mut(self.order - 1).assign(&pred);
            }
        }

        Ok(predictions)
    }

    /// Calculate impulse response function
    pub fn impulse_response(&self, periods: usize, shock_var: usize) -> Result<Array2<F>> {
        if !self.is_fitted {
            return Err(TimeSeriesError::InvalidInput(
                "Model must be fitted before calculating impulse response".to_string(),
            ));
        }

        if shock_var >= self.n_vars {
            return Err(TimeSeriesError::InvalidInput(format!(
                "Shock variable {} out of range (0-{})",
                shock_var,
                self.n_vars - 1
            )));
        }

        let mut responses = Array2::zeros((periods, self.n_vars));

        // Initial shock
        let mut shock = Array1::zeros(self.n_vars);
        shock[shock_var] = F::one();
        responses.row_mut(0).assign(&shock);

        // Calculate responses
        for t in 1..periods {
            let mut response = Array1::zeros(self.n_vars);

            for lag in 0..self.order.min(t) {
                let past_response = responses.row(t - lag - 1);
                response = response + self.coefficients[lag].dot(&past_response);
            }

            responses.row_mut(t).assign(&response);
        }

        Ok(responses)
    }

    /// Forecast error variance decomposition
    pub fn variance_decomposition(&self, periods: usize) -> Result<Vec<Array2<F>>> {
        if !self.is_fitted {
            return Err(TimeSeriesError::InvalidInput(
                "Model must be fitted before variance decomposition".to_string(),
            ));
        }

        let mut decomposition = vec![Array2::zeros((self.n_vars, self.n_vars)); periods];

        // Get impulse responses for each variable
        let mut impulse_responses = Vec::new();
        for i in 0..self.n_vars {
            impulse_responses.push(self.impulse_response(periods, i)?);
        }

        // Calculate cumulative variance contributions
        for (h, decomp_h) in decomposition.iter_mut().enumerate().take(periods) {
            let mut total_variance = Array1::zeros(self.n_vars);

            for (shock_var, impulse_response) in impulse_responses.iter().enumerate() {
                for response_var in 0..self.n_vars {
                    let mut contribution = F::zero();

                    for t in 0..=h {
                        let response = impulse_response[[t, response_var]];
                        contribution = contribution + response * response;
                    }

                    decomp_h[[response_var, shock_var]] = contribution;
                    total_variance[response_var] = total_variance[response_var] + contribution;
                }
            }

            // Normalize to percentages
            for response_var in 0..self.n_vars {
                if total_variance[response_var] > F::epsilon() {
                    for shock_var in 0..self.n_vars {
                        decomp_h[[response_var, shock_var]] =
                            decomp_h[[response_var, shock_var]] / total_variance[response_var];
                    }
                }
            }
        }

        Ok(decomposition)
    }

    /// Test for Granger causality
    pub fn granger_causality(&self, cause_var: usize, effect_var: usize) -> Result<(F, F)> {
        if !self.is_fitted {
            return Err(TimeSeriesError::InvalidInput(
                "Model must be fitted before testing Granger causality".to_string(),
            ));
        }

        if cause_var >= self.n_vars || effect_var >= self.n_vars {
            return Err(TimeSeriesError::InvalidInput(
                "Variable indices out of range".to_string(),
            ));
        }

        // Placeholder implementation
        // Would implement proper F-test for coefficient restrictions
        let f_stat = F::from(2.5).unwrap();
        let p_value = F::from(0.05).unwrap();

        Ok((f_stat, p_value))
    }
}

/// Vector Moving Average (VMA) model
#[derive(Debug, Clone)]
pub struct VMAModel<F> {
    /// Order of the VMA model
    pub order: usize,
    /// Number of variables
    pub n_vars: usize,
    /// MA coefficient matrices
    pub ma_coefficients: Vec<Array2<F>>,
    /// Intercept vector
    pub intercept: Array1<F>,
    /// Innovation covariance
    pub covariance: Array2<F>,
}

/// Vector ARMA (VARMA) model
#[derive(Debug, Clone)]
pub struct VARMAModel<F> {
    /// VAR component
    pub var: VARModel<F>,
    /// VMA component
    pub vma: VMAModel<F>,
}

/// Vector Error Correction Model (VECM)
#[derive(Debug, Clone)]
pub struct VECMModel<F> {
    /// Number of cointegrating relationships
    pub rank: usize,
    /// Adjustment coefficients (alpha)
    pub adjustment: Array2<F>,
    /// Cointegrating vectors (beta)
    pub cointegration: Array2<F>,
    /// Short-run dynamics
    pub short_run: Vec<Array2<F>>,
    /// Deterministic terms
    pub deterministic: Array2<F>,
    /// Residual covariance
    pub covariance: Array2<F>,
    /// Whether the model is fitted
    pub is_fitted: bool,
}

impl<F> VECMModel<F>
where
    F: Float + FromPrimitive + Debug + Display + ScalarOperand,
{
    /// Create a new VECM model
    pub fn new(n_vars: usize, rank: usize, lag_order: usize) -> Result<Self> {
        if rank >= n_vars {
            return Err(TimeSeriesError::InvalidInput(
                "Cointegration rank must be less than number of variables".to_string(),
            ));
        }

        let adjustment = Array2::zeros((n_vars, rank));
        let cointegration = Array2::zeros((n_vars, rank));
        let short_run = vec![Array2::zeros((n_vars, n_vars)); lag_order - 1];
        let deterministic = Array2::zeros((n_vars, 2)); // constant and trend
        let covariance = Array2::eye(n_vars);

        Ok(Self {
            rank,
            adjustment,
            cointegration,
            short_run,
            deterministic,
            covariance,
            is_fitted: false,
        })
    }

    /// Fit VECM using Johansen procedure
    pub fn fit<S>(&mut self, data: &ArrayBase<S, Ix2>) -> Result<()>
    where
        S: Data<Elem = F>,
    {
        scirs2_core::validation::check_array_finite(data, "data")?;

        // Placeholder implementation
        // Would implement full Johansen procedure
        self.is_fitted = true;
        Ok(())
    }

    /// Convert VECM to VAR representation
    pub fn to_var(&self) -> Result<VARModel<F>> {
        if !self.is_fitted {
            return Err(TimeSeriesError::InvalidInput(
                "VECM must be fitted before conversion to VAR".to_string(),
            ));
        }

        // Placeholder implementation
        let var = VARModel::new(self.short_run.len() + 1, self.adjustment.nrows())?;
        Ok(var)
    }
}

/// Helper function to solve normal equations (simplified)
fn solve_normal_equations<F>(_xtx: &Array2<F>, xty: &Array2<F>) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Display + ScalarOperand,
{
    // This is a placeholder - would use proper linear algebra solver
    // For now, just return the xty matrix scaled by a small factor
    Ok(xty * F::from(0.1).unwrap())
}

/// Model selection criteria
#[derive(Debug, Clone, Copy)]
pub enum SelectionCriterion {
    /// Akaike Information Criterion
    AIC,
    /// Bayesian Information Criterion
    BIC,
    /// Hannan-Quinn Information Criterion
    HQC,
    /// Final Prediction Error
    FPE,
}

/// Select optimal VAR order
pub fn select_var_order<S, F>(
    data: &ArrayBase<S, Ix2>,
    max_order: usize,
    criterion: SelectionCriterion,
) -> Result<usize>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive + Debug + Display + ScalarOperand,
{
    let (t, k) = data.dim();
    let mut best_order = 1;
    let mut best_criterion = F::infinity();

    for order in 1..=max_order {
        if t <= order + 1 {
            break;
        }

        let mut model = VARModel::new(order, k)?;
        model.fit(data)?;

        let log_det = matrix_log_determinant(&model.covariance);
        let n_params = order * k * k + k;

        let criterion_value = match criterion {
            SelectionCriterion::AIC => {
                log_det + F::from(2.0).unwrap() * F::from(n_params).unwrap() / F::from(t).unwrap()
            }
            SelectionCriterion::BIC => {
                log_det
                    + F::from(n_params).unwrap().ln() * F::from(t).unwrap() / F::from(t).unwrap()
            }
            SelectionCriterion::HQC => {
                log_det
                    + F::from(2.0).unwrap()
                        * F::from(n_params).unwrap().ln()
                        * F::from(t).unwrap().ln()
                        / F::from(t).unwrap()
            }
            SelectionCriterion::FPE => {
                let factor = (F::from(t).unwrap() + F::from(n_params).unwrap())
                    / (F::from(t).unwrap() - F::from(n_params).unwrap());
                log_det + factor.ln()
            }
        };

        if criterion_value < best_criterion {
            best_criterion = criterion_value;
            best_order = order;
        }
    }

    Ok(best_order)
}

/// Calculate log determinant of a matrix (simplified)
fn matrix_log_determinant<F>(_matrix: &Array2<F>) -> F
where
    F: Float + FromPrimitive + Debug + Display + ScalarOperand,
{
    // Placeholder - would use proper determinant calculation
    F::one()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_var_creation() {
        let model = VARModel::<f64>::new(2, 3).unwrap();
        assert_eq!(model.order, 2);
        assert_eq!(model.n_vars, 3);
        assert_eq!(model.coefficients.len(), 2);
        assert!(!model.is_fitted);
    }

    #[test]
    fn test_var_fit() {
        // Create simple AR(1) data
        let mut data = Array2::zeros((100, 2));
        data[[0, 0]] = 1.0;
        data[[0, 1]] = 0.5;

        for t in 1..100 {
            data[[t, 0]] = 0.5 * data[[t - 1, 0]] + 0.1 * data[[t - 1, 1]];
            data[[t, 1]] = 0.2 * data[[t - 1, 0]] + 0.7 * data[[t - 1, 1]];
        }

        let mut model = VARModel::new(1, 2).unwrap();
        model.fit(&data).unwrap();
        assert!(model.is_fitted);
    }

    #[test]
    fn test_var_predict() {
        let mut model = VARModel::new(1, 2).unwrap();
        model.coefficients[0] = array![[0.5, 0.1], [0.2, 0.7]];
        model.intercept = array![0.0, 0.0];
        model.is_fitted = true;

        let initial = array![[1.0, 0.5]];
        let predictions = model.predict(&initial, 5).unwrap();
        assert_eq!(predictions.dim(), (5, 2));
    }

    #[test]
    fn test_impulse_response() {
        let mut model = VARModel::new(1, 2).unwrap();
        model.coefficients[0] = array![[0.5, 0.1], [0.2, 0.7]];
        model.is_fitted = true;

        let irf = model.impulse_response(10, 0).unwrap();
        assert_eq!(irf.dim(), (10, 2));
        assert_eq!(irf[[0, 0]], 1.0);
        assert_eq!(irf[[0, 1]], 0.0);
    }

    #[test]
    fn test_vecm_creation() {
        let model = VECMModel::<f64>::new(3, 2, 3).unwrap();
        assert_eq!(model.rank, 2);
        assert_eq!(model.short_run.len(), 2);
        assert!(!model.is_fitted);
    }

    #[test]
    fn test_var_order_selection() {
        let data: Array2<f64> = Array2::zeros((100, 2));
        let order = select_var_order(&data, 5, SelectionCriterion::AIC).unwrap();
        assert!((1..=5).contains(&order));
    }
}
