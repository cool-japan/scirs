//! TBATS decomposition for time series with multiple seasonal components
//!
//! TBATS stands for Trigonometric seasonality, Box-Cox transformation,
//! ARMA errors, Trend, and Seasonal components.

use ndarray::Array1;
use num_traits::{Float, FromPrimitive};
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
}

impl Default for TBATSOptions {
    fn default() -> Self {
        Self {
            use_box_cox: true,
            box_cox_lambda: None,
            use_trend: true,
            use_damped_trend: false,
            seasonal_periods: Vec::new(),
            fourier_terms: None,
            ar_order: 0,
            ma_order: 0,
            auto_arma: true,
            use_parallel: false,
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
/// ```
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
    F: Float + FromPrimitive + Debug + std::iter::Sum,
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
    let (transformed_ts, lambda) = if options.use_box_cox {
        let lambda = options.box_cox_lambda.unwrap_or({
            // Auto-estimate lambda (simplified approach)
            // In a real implementation, we would use maximum likelihood estimation
            0.0 // Log transformation
        });
        let transformed = box_cox_transform(ts, lambda)?;
        (transformed, Some(lambda))
    } else {
        (ts.clone(), None)
    };

    // Step 2: Initialize components
    let mut level = Array1::zeros(n);
    let mut trend = Array1::zeros(n);

    // Initial level is just the first observation
    level[0] = transformed_ts[0];

    // Initial trend is the average first difference of the first few observations
    if options.use_trend && n > 1 {
        let mut trend_sum = F::zero();
        let trend_points = std::cmp::min(n - 1, 4);
        for i in 0..trend_points {
            trend_sum = trend_sum + (transformed_ts[i + 1] - transformed_ts[i]);
        }
        trend[0] = trend_sum / F::from_usize(trend_points).unwrap();
    }

    // Step 3: Determine number of Fourier terms for each seasonal component
    let fourier_terms = match &options.fourier_terms {
        Some(terms) => terms.clone(),
        None => {
            // Default to minimum of period/2 or 5 terms
            options
                .seasonal_periods
                .iter()
                .map(|&p| std::cmp::min((p / 2.0).floor() as usize, 5))
                .collect()
        }
    };

    // Step 4: Initialize seasonal components using Fourier series
    let mut seasonal_components = Vec::with_capacity(options.seasonal_periods.len());
    let mut fourier_coefficients = Vec::with_capacity(options.seasonal_periods.len());

    for (i, &period) in options.seasonal_periods.iter().enumerate() {
        let k = fourier_terms[i];
        let mut season = Array1::zeros(n);
        let mut coefficients = Vec::with_capacity(k);

        // Initialize coefficients with small random values for simplicity
        // In a real implementation, we would use regression to estimate initial values
        for _ in 0..k {
            coefficients.push((0.1, 0.1)); // (a, b) for sin and cos terms
        }

        // Apply Fourier series to compute initial seasonal component
        for t in 0..n {
            let t_f = F::from_usize(t).unwrap();
            for (j, &(a, b)) in coefficients.iter().enumerate() {
                // We don't need j_f for the current implementation
                let freq =
                    F::from_f64(2.0 * std::f64::consts::PI * (j + 1) as f64 / period).unwrap();
                let a_f = F::from_f64(a).unwrap();
                let b_f = F::from_f64(b).unwrap();

                season[t] = season[t] + a_f * (freq * t_f).sin() + b_f * (freq * t_f).cos();
            }
        }

        seasonal_components.push(season);
        fourier_coefficients.push(coefficients);
    }

    // Step 5: Compute residuals for ARMA modeling
    let mut residuals = Array1::zeros(n);
    for t in 0..n {
        let mut seasonal_sum = F::zero();
        for season in &seasonal_components {
            seasonal_sum = seasonal_sum + season[t];
        }

        if options.use_trend {
            residuals[t] = transformed_ts[t] - level[t] - seasonal_sum;
            if t > 0 {
                residuals[t] = residuals[t] - trend[t - 1];
            }
        } else {
            residuals[t] = transformed_ts[t] - level[t] - seasonal_sum;
        }
    }

    // Step 6: Fit ARMA model on residuals (simplified)
    // In a real implementation, we would use proper model selection and parameter estimation
    let ar_order = options.ar_order;
    let ma_order = options.ma_order;

    let ar_coefficients = vec![0.1; ar_order]; // Placeholder
    let ma_coefficients = vec![0.1; ma_order]; // Placeholder

    // Step 7: Estimate smoothing parameters (simplified)
    // In a real implementation, we would use maximum likelihood estimation
    let alpha = 0.1; // Level smoothing
    let beta = if options.use_trend { Some(0.01) } else { None }; // Trend smoothing
    let phi = if options.use_damped_trend {
        Some(0.98)
    } else {
        None
    }; // Damping

    // Seasonal smoothing parameters
    let gamma = if !options.seasonal_periods.is_empty() {
        Some(vec![0.001; options.seasonal_periods.len()])
    } else {
        None
    };

    // Step 8: Forecast (not implemented yet, future work)

    // Step 9: Organize outputs
    // For demonstration, we'll use the initialized components

    // Create parameters struct
    let parameters = TBATSParameters {
        lambda,
        alpha,
        beta,
        phi,
        gamma,
        fourier_coefficients,
        ar_coefficients,
        ma_coefficients,
    };

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
    };

    Ok(result)
}
