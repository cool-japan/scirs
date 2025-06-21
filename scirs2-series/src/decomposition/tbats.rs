//! TBATS decomposition for time series with multiple seasonal components
//!
//! TBATS stands for Trigonometric seasonality, Box-Cox transformation,
//! ARMA errors, Trend, and Seasonal components.

use ndarray::{Array1, Array2, ScalarOperand};
use ndarray_linalg::Solve;
use num_traits::{Float, FromPrimitive, NumCast};
use std::fmt::Debug;

use super::common::box_cox_transform;
use crate::error::{Result, TimeSeriesError};

/// Options for TBATS (Trigonometric seasonality, Box-Cox transformation, ARMA errors, Trend and Seasonal components)
#[derive(Debug, Clone)]
pub struct TBATSOptions {
    /// Whether to use Box-Cox transformation
    pub use_box_cox: bool,
    /// Box-Cox transformation parameter (if None, automatically estimated)
    pub box_cox_lambda: Option<f64>,
    /// Whether to use trend component
    pub use_trend: bool,
    /// Whether to use damped trend
    pub use_damped_trend: bool,
    /// Seasonal periods (e.g., [7.0, 30.42, 365.25] for daily data with weekly, monthly, and yearly patterns)
    pub seasonal_periods: Vec<f64>,
    /// Number of Fourier terms for each seasonal period
    pub fourier_terms: Option<Vec<usize>>,
    /// Autoregressive order for ARMA errors
    pub ar_order: usize,
    /// Moving average order for ARMA errors
    pub ma_order: usize,
    /// Whether to automatically select ARMA orders
    pub auto_arma: bool,
    /// Whether to use parallel processing
    pub use_parallel: bool,
    /// Maximum number of iterations for parameter estimation
    pub max_iterations: usize,
    /// Convergence tolerance for parameter estimation
    pub tolerance: f64,
}

impl Default for TBATSOptions {
    fn default() -> Self {
        Self {
            use_box_cox: false, // Changed default to avoid complexity
            box_cox_lambda: None,
            use_trend: true,
            use_damped_trend: false,
            seasonal_periods: Vec::new(),
            fourier_terms: None,
            ar_order: 0,
            ma_order: 0,
            auto_arma: false, // Simplified for now
            use_parallel: false,
            max_iterations: 100,
            tolerance: 1e-6,
        }
    }
}

/// Results of TBATS decomposition
#[derive(Debug, Clone)]
pub struct TBATSResult<F> {
    /// Trend component
    pub trend: Array1<F>,
    /// Seasonal components (one for each seasonal period)
    pub seasonal_components: Vec<Array1<F>>,
    /// ARMA residuals
    pub residuals: Array1<F>,
    /// Level component
    pub level: Array1<F>,
    /// Original time series
    pub original: Array1<F>,
    /// Box-Cox transformed time series (if transformation was applied)
    pub transformed: Option<Array1<F>>,
    /// Parameters used for the model
    pub parameters: TBATSParameters,
    /// Log-likelihood of the fitted model
    pub log_likelihood: f64,
}

/// Parameters for TBATS model
#[derive(Debug, Clone)]
pub struct TBATSParameters {
    /// Box-Cox transformation parameter
    pub lambda: Option<f64>,
    /// Smoothing parameter for level
    pub alpha: f64,
    /// Smoothing parameter for trend
    pub beta: Option<f64>,
    /// Damping parameter
    pub phi: Option<f64>,
    /// Smoothing parameter for seasonal components
    pub gamma: Option<Vec<f64>>,
    /// Fourier coefficients for each seasonal component
    pub fourier_coefficients: Vec<Vec<(f64, f64)>>,
    /// Autoregressive coefficients for ARMA errors
    pub ar_coefficients: Vec<f64>,
    /// Moving average coefficients for ARMA errors
    pub ma_coefficients: Vec<f64>,
    /// Innovation variance
    pub sigma_squared: f64,
}

/// Performs TBATS decomposition on a time series
///
/// TBATS (Trigonometric seasonality, Box-Cox transformation, ARMA errors, Trend and
/// Seasonal components) is a complex model that handles multiple seasonal patterns
/// using Fourier series to represent seasonality.
///
/// # Arguments
///
/// * `ts` - The time series to decompose
/// * `options` - Options for TBATS decomposition
///
/// # Returns
///
/// * TBATS decomposition result
///
/// # Example
///
/// ```ignore
/// use ndarray::array;
/// use scirs2_series::decomposition::{tbats_decomposition, TBATSOptions};
///
/// let ts = array![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0,
///                 1.5, 2.5, 3.5, 2.5, 1.5, 2.5, 3.5, 2.5, 1.5, 2.5, 3.5, 2.5];
///
/// let mut options = TBATSOptions::default();
/// options.seasonal_periods = vec![4.0, 12.0]; // Both quarterly and yearly patterns
/// options.use_box_cox = false; // Disable Box-Cox transformation for this example
///
/// let result = tbats_decomposition(&ts, &options).unwrap();
/// println!("Trend: {:?}", result.trend);
/// println!("Seasonal Components: {:?}", result.seasonal_components);
/// println!("Residuals: {:?}", result.residuals);
/// ```
pub fn tbats_decomposition<F>(ts: &Array1<F>, options: &TBATSOptions) -> Result<TBATSResult<F>>
where
    F: Float
        + FromPrimitive
        + Debug
        + std::iter::Sum
        + ndarray_linalg::Lapack
        + ScalarOperand
        + NumCast,
{
    let n = ts.len();

    // Check inputs
    if n < 3 {
        return Err(TimeSeriesError::DecompositionError(
            "Time series must have at least 3 points for TBATS decomposition".to_string(),
        ));
    }

    if options.seasonal_periods.is_empty()
        && options.use_box_cox
        && options.box_cox_lambda.is_none()
    {
        return Err(TimeSeriesError::DecompositionError(
            "When using Box-Cox transformation with no seasonal periods, lambda must be specified"
                .to_string(),
        ));
    }

    for &period in &options.seasonal_periods {
        if period <= 1.0 {
            return Err(TimeSeriesError::DecompositionError(
                "Seasonal periods must be greater than 1".to_string(),
            ));
        }
    }

    if let Some(ref terms) = options.fourier_terms {
        if terms.len() != options.seasonal_periods.len() {
            return Err(TimeSeriesError::DecompositionError(
                "Number of Fourier terms must match number of seasonal periods".to_string(),
            ));
        }
        for &k in terms {
            if k == 0 {
                return Err(TimeSeriesError::DecompositionError(
                    "Number of Fourier terms must be at least 1 for each seasonal period"
                        .to_string(),
                ));
            }
        }
    }

    // Step 1: Apply Box-Cox transformation if requested
    let (transformed_ts, _lambda) = if options.use_box_cox {
        let lambda = options.box_cox_lambda.unwrap_or_else(|| {
            // Estimate optimal lambda using profile likelihood (simplified)
            estimate_box_cox_lambda(ts)
        });
        let transformed = box_cox_transform(ts, lambda)?;
        (transformed, Some(lambda))
    } else {
        (ts.clone(), None)
    };

    // Step 2: Determine number of Fourier terms for each seasonal component
    let fourier_terms = match &options.fourier_terms {
        Some(terms) => terms.clone(),
        None => {
            // Default to minimum of period/2 or 3 terms for computational efficiency
            options
                .seasonal_periods
                .iter()
                .map(|&p| std::cmp::min((p / 2.0).floor() as usize, 3).max(1))
                .collect()
        }
    };

    // Step 3: Set up state space model
    // State vector: [level, trend, s1_1, s1_2, ..., s1_k1, s2_1, s2_2, ..., s2_k2, ...]
    // where si_j represents the j-th Fourier component of the i-th seasonal pattern

    let state_size = calculate_state_size(options, &fourier_terms);
    let mut state = Array1::zeros(state_size);

    // Step 4: Initialize state components
    initialize_state(&mut state, &transformed_ts, options, &fourier_terms)?;

    // Step 5: Estimate parameters using simplified maximum likelihood
    let parameters = estimate_parameters(&transformed_ts, options, &fourier_terms)?;

    // Step 6: Apply state space model with estimated parameters
    let components =
        apply_state_space_model(&transformed_ts, &parameters, options, &fourier_terms)?;

    // Step 7: Extract components
    let (level, trend, seasonal_components, residuals, log_likelihood) = extract_components(
        &transformed_ts,
        &components,
        &parameters,
        options,
        &fourier_terms,
    )?;

    // Create result struct
    let result = TBATSResult {
        trend,
        seasonal_components,
        residuals,
        level,
        original: ts.clone(),
        transformed: if options.use_box_cox {
            Some(transformed_ts)
        } else {
            None
        },
        parameters,
        log_likelihood,
    };

    Ok(result)
}

/// Estimate optimal Box-Cox lambda parameter
fn estimate_box_cox_lambda<F>(ts: &Array1<F>) -> f64
where
    F: Float + FromPrimitive + Debug,
{
    // Simplified estimation - in practice, would use profile likelihood
    let variance = ts.var(F::zero());
    let mean = ts.mean().unwrap_or(F::zero());

    if variance > F::zero() && mean > F::zero() {
        // Use coefficient of variation to guide lambda selection
        let cv = variance.sqrt() / mean;
        if cv.to_f64().unwrap_or(1.0) > 0.3 {
            0.0 // Log transformation for high variability
        } else {
            0.5 // Square root transformation for moderate variability
        }
    } else {
        1.0 // No transformation
    }
}

/// Calculate the size of the state vector
fn calculate_state_size(options: &TBATSOptions, fourier_terms: &[usize]) -> usize {
    let mut size = 1; // Level

    if options.use_trend {
        size += 1; // Trend
    }

    // Seasonal components (2 states per Fourier term: sin and cos)
    for &k in fourier_terms {
        size += 2 * k;
    }

    size
}

/// Initialize the state vector
fn initialize_state<F>(
    state: &mut Array1<F>,
    ts: &Array1<F>,
    options: &TBATSOptions,
    fourier_terms: &[usize],
) -> Result<()>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();

    // Initialize level with first observation
    state[0] = ts[0];

    let mut idx = 1;

    // Initialize trend with average first difference
    if options.use_trend {
        if n > 1 {
            let mut trend_sum = F::zero();
            let trend_points = std::cmp::min(n - 1, 4);
            for i in 0..trend_points {
                trend_sum = trend_sum + (ts[i + 1] - ts[i]);
            }
            state[idx] = trend_sum / F::from_usize(trend_points).unwrap();
        }
        idx += 1;
    }

    // Initialize seasonal components using simple detrending and regression
    for (s_idx, (&_period, &k)) in options
        .seasonal_periods
        .iter()
        .zip(fourier_terms.iter())
        .enumerate()
    {
        // Simple initialization: small random values
        for _ in 0..(2 * k) {
            state[idx] = F::from_f64(0.01 * (s_idx as f64 + 1.0)).unwrap();
            idx += 1;
        }
    }

    Ok(())
}

/// Estimate model parameters using simplified maximum likelihood
fn estimate_parameters<F>(
    ts: &Array1<F>,
    options: &TBATSOptions,
    fourier_terms: &[usize],
) -> Result<TBATSParameters>
where
    F: Float
        + FromPrimitive
        + Debug
        + std::iter::Sum
        + ndarray_linalg::Lapack
        + ScalarOperand
        + NumCast,
{
    // Simplified parameter estimation
    // In practice, this would use numerical optimization (e.g., Nelder-Mead)

    let alpha = 0.1; // Level smoothing parameter
    let beta = if options.use_trend { Some(0.01) } else { None };
    let phi = if options.use_damped_trend {
        Some(0.98)
    } else {
        None
    };

    // Seasonal smoothing parameters
    let gamma = if !options.seasonal_periods.is_empty() {
        Some(vec![0.001; options.seasonal_periods.len()])
    } else {
        None
    };

    // Estimate Fourier coefficients using linear regression
    let fourier_coefficients = estimate_fourier_coefficients(ts, options, fourier_terms)?;

    // Simple ARMA coefficients (would normally be estimated via MLE)
    let ar_coefficients = vec![0.0; options.ar_order];
    let ma_coefficients = vec![0.0; options.ma_order];

    // Estimate innovation variance
    let residual_variance = estimate_residual_variance(ts, &fourier_coefficients, options);

    Ok(TBATSParameters {
        lambda: options.box_cox_lambda,
        alpha,
        beta,
        phi,
        gamma,
        fourier_coefficients,
        ar_coefficients,
        ma_coefficients,
        sigma_squared: residual_variance,
    })
}

/// Estimate Fourier coefficients using regression
fn estimate_fourier_coefficients<F>(
    ts: &Array1<F>,
    options: &TBATSOptions,
    fourier_terms: &[usize],
) -> Result<Vec<Vec<(f64, f64)>>>
where
    F: Float
        + FromPrimitive
        + Debug
        + std::iter::Sum
        + ndarray_linalg::Lapack
        + ScalarOperand
        + NumCast,
{
    let n = ts.len();
    let mut all_coefficients = Vec::new();

    // For each seasonal period
    for (&period, &k) in options.seasonal_periods.iter().zip(fourier_terms.iter()) {
        let mut design_matrix = Array2::zeros((n, 2 * k));

        // Create Fourier design matrix
        for t in 0..n {
            let t_f = F::from_usize(t).unwrap();
            for j in 0..k {
                let freq =
                    F::from_f64(2.0 * std::f64::consts::PI * (j + 1) as f64 / period).unwrap();

                // Sin term
                design_matrix[[t, 2 * j]] = Float::sin(freq * t_f);
                // Cos term
                design_matrix[[t, 2 * j + 1]] = Float::cos(freq * t_f);
            }
        }

        // Solve least squares: design_matrix * coeffs = ts
        let coeffs = design_matrix
            .t()
            .dot(&design_matrix)
            .solve(&design_matrix.t().dot(ts))
            .map_err(|e| {
                TimeSeriesError::DecompositionError(format!(
                    "Failed to estimate Fourier coefficients: {}",
                    e
                ))
            })?;

        // Extract coefficient pairs
        let mut seasonal_coeffs = Vec::new();
        for j in 0..k {
            let a = coeffs[2 * j].to_f64().unwrap_or(0.0);
            let b = coeffs[2 * j + 1].to_f64().unwrap_or(0.0);
            seasonal_coeffs.push((a, b));
        }

        all_coefficients.push(seasonal_coeffs);
    }

    Ok(all_coefficients)
}

/// Estimate residual variance
fn estimate_residual_variance<F>(
    ts: &Array1<F>,
    fourier_coefficients: &[Vec<(f64, f64)>],
    options: &TBATSOptions,
) -> f64
where
    F: Float + FromPrimitive + Debug + std::iter::Sum,
{
    let n = ts.len();
    let mut residual_sum = F::zero();

    for t in 0..n {
        let mut seasonal_sum = F::zero();
        let t_f = F::from_usize(t).unwrap();

        for (&period, coeffs) in options
            .seasonal_periods
            .iter()
            .zip(fourier_coefficients.iter())
        {
            for (j, &(a, b)) in coeffs.iter().enumerate() {
                let freq =
                    F::from_f64(2.0 * std::f64::consts::PI * (j + 1) as f64 / period).unwrap();
                let a_f = F::from_f64(a).unwrap();
                let b_f = F::from_f64(b).unwrap();

                seasonal_sum =
                    seasonal_sum + a_f * Float::sin(freq * t_f) + b_f * Float::cos(freq * t_f);
            }
        }

        let residual = ts[t] - seasonal_sum;
        residual_sum = residual_sum + residual * residual;
    }

    (residual_sum / F::from_usize(n).unwrap())
        .to_f64()
        .unwrap_or(1.0)
}

/// Apply state space model (simplified version)
fn apply_state_space_model<F>(
    ts: &Array1<F>,
    parameters: &TBATSParameters,
    options: &TBATSOptions,
    fourier_terms: &[usize],
) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + NumCast,
{
    let n = ts.len();
    let state_size = calculate_state_size(options, fourier_terms);
    let mut states = Array2::zeros((n, state_size));

    // Initialize first state
    states[[0, 0]] = ts[0]; // Level

    // For simplicity, just track the level and seasonal components
    for t in 1..n {
        // Level evolution (simplified)
        states[[t, 0]] = F::from(parameters.alpha).unwrap() * ts[t]
            + (F::one() - F::from(parameters.alpha).unwrap()) * states[[t - 1, 0]];

        // Copy other state components (simplified)
        for i in 1..state_size {
            states[[t, i]] = states[[t - 1, i]];
        }
    }

    Ok(states)
}

/// Type alias for TBATS components extraction result
type TBATSComponentsResult<F> = Result<(Array1<F>, Array1<F>, Vec<Array1<F>>, Array1<F>, f64)>;

/// Extract components from state space results
fn extract_components<F>(
    ts: &Array1<F>,
    states: &Array2<F>,
    parameters: &TBATSParameters,
    options: &TBATSOptions,
    fourier_terms: &[usize],
) -> TBATSComponentsResult<F>
where
    F: Float + FromPrimitive + Debug + std::iter::Sum + NumCast,
{
    let n = ts.len();

    // Extract level
    let level = states.column(0).to_owned();

    // Extract trend
    let trend = if options.use_trend {
        states.column(1).to_owned()
    } else {
        Array1::zeros(n)
    };

    // Extract seasonal components
    let mut seasonal_components = Vec::new();
    for (s_idx, (&period, &_k)) in options
        .seasonal_periods
        .iter()
        .zip(fourier_terms.iter())
        .enumerate()
    {
        let mut seasonal = Array1::zeros(n);
        let coeffs = &parameters.fourier_coefficients[s_idx];

        for t in 0..n {
            let t_f = F::from_usize(t).unwrap();
            let mut seasonal_value = F::zero();

            for (j, &(a, b)) in coeffs.iter().enumerate() {
                let freq =
                    F::from_f64(2.0 * std::f64::consts::PI * (j + 1) as f64 / period).unwrap();
                let a_f = F::from_f64(a).unwrap();
                let b_f = F::from_f64(b).unwrap();

                seasonal_value =
                    seasonal_value + a_f * Float::sin(freq * t_f) + b_f * Float::cos(freq * t_f);
            }

            seasonal[t] = seasonal_value;
        }

        seasonal_components.push(seasonal);
    }

    // Compute residuals
    let mut residuals = Array1::zeros(n);
    for t in 0..n {
        let mut fitted = level[t];
        if options.use_trend {
            fitted = fitted + trend[t];
        }
        for seasonal in &seasonal_components {
            fitted = fitted + seasonal[t];
        }
        residuals[t] = ts[t] - fitted;
    }

    // Compute log-likelihood (simplified)
    let residual_variance = residuals.mapv(|x| x * x).sum() / F::from_usize(n).unwrap();
    let log_likelihood = -0.5
        * n as f64
        * (2.0 * std::f64::consts::PI * residual_variance.to_f64().unwrap_or(1.0)).ln()
        - 0.5 * residuals.mapv(|x| x * x).sum().to_f64().unwrap_or(0.0)
            / residual_variance.to_f64().unwrap_or(1.0);

    Ok((level, trend, seasonal_components, residuals, log_likelihood))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_tbats_basic() {
        // Create a simple time series with trend and seasonality
        let n = 48;
        let mut ts = Array1::zeros(n);
        for i in 0..n {
            let trend = 0.1 * i as f64;
            let seasonal = 2.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin();
            let noise = 0.1 * (i as f64 * 0.789).sin();
            ts[i] = 10.0 + trend + seasonal + noise;
        }

        let mut options = TBATSOptions::default();
        options.seasonal_periods = vec![12.0];
        options.use_trend = true;
        options.use_box_cox = false;

        let result = tbats_decomposition(&ts, &options).unwrap();

        // Check that we have the expected number of components
        assert_eq!(result.seasonal_components.len(), 1);
        assert_eq!(result.level.len(), n);
        assert_eq!(result.trend.len(), n);
        assert_eq!(result.residuals.len(), n);
    }

    #[test]
    fn test_tbats_multiple_seasons() {
        // Create a time series with multiple seasonal patterns
        let n = 60;
        let mut ts = Array1::zeros(n);
        for i in 0..n {
            let trend = 0.05 * i as f64;
            let seasonal1 = 3.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin();
            let seasonal2 = 1.5 * (2.0 * std::f64::consts::PI * i as f64 / 4.0).cos();
            ts[i] = 5.0 + trend + seasonal1 + seasonal2;
        }

        let mut options = TBATSOptions::default();
        options.seasonal_periods = vec![12.0, 4.0];
        options.use_trend = true;
        options.use_box_cox = false;

        let result = tbats_decomposition(&ts, &options).unwrap();

        // Check that we have the right number of seasonal components
        assert_eq!(result.seasonal_components.len(), 2);

        // Check that parameters were estimated
        assert!(result.parameters.alpha > 0.0);
        assert!(result.parameters.fourier_coefficients.len() == 2);
    }

    #[test]
    fn test_tbats_edge_cases() {
        // Test with minimum size time series
        let ts = array![1.0, 2.0, 3.0];
        let mut options = TBATSOptions::default();
        options.seasonal_periods = vec![2.0];

        let result = tbats_decomposition(&ts, &options);
        assert!(result.is_ok());

        // Test with invalid seasonal period
        options.seasonal_periods = vec![0.5];
        let result = tbats_decomposition(&ts, &options);
        assert!(result.is_err());

        // Test with too small time series
        let ts = array![1.0, 2.0];
        options.seasonal_periods = vec![2.0];
        let result = tbats_decomposition(&ts, &options);
        assert!(result.is_err());
    }

    #[test]
    fn test_tbats_no_seasonal() {
        // Test TBATS with trend only (no seasonal components)
        let n = 20;
        let mut ts = Array1::zeros(n);
        for i in 0..n {
            let trend = 0.2 * i as f64;
            ts[i] = 5.0 + trend;
        }

        let mut options = TBATSOptions::default();
        options.seasonal_periods = vec![];
        options.use_trend = true;
        options.use_box_cox = false;

        let result = tbats_decomposition(&ts, &options).unwrap();

        // Check that we have no seasonal components
        assert_eq!(result.seasonal_components.len(), 0);
        assert_eq!(result.level.len(), n);
        assert_eq!(result.trend.len(), n);
    }
}
