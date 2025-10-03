//! Volatility estimators for financial time series
//!
//! This module provides various volatility estimation techniques used in
//! financial econometrics. Each estimator has different properties and
//! use cases depending on the available data and required accuracy.
//!
//! # Overview
//!
//! Volatility estimation is crucial in financial modeling for:
//! - Risk management and VaR calculations
//! - Option pricing and hedging
//! - Portfolio optimization
//! - Regulatory capital requirements
//!
//! # Categories
//!
//! ## High-Frequency Estimators
//! These use intraday high, low, open, close (HLOC) data:
//! - **Realized Volatility**: Sum of squared returns
//! - **Garman-Klass**: Uses HLOC, most efficient unbiased estimator
//! - **Parkinson**: Uses only high and low prices
//! - **Rogers-Satchell**: Drift-independent, uses HLOC
//! - **Yang-Zhang**: Handles opening gaps and overnight returns
//!
//! ## Time Series Models
//! These model volatility evolution over time:
//! - **GARCH**: Simple GARCH(1,1) volatility estimation
//! - **EWMA**: Exponentially weighted moving average
//!
//! ## Range-Based Estimators
//! These use price ranges for efficiency:
//! - **Range Volatility**: Uses high-low ranges over periods
//! - **Intraday Volatility**: Sampling-based intraday estimation
//!
//! # Efficiency Comparison
//!
//! Based on theoretical efficiency (lower is better):
//! 1. Garman-Klass (most efficient)
//! 2. Rogers-Satchell  
//! 3. Yang-Zhang
//! 4. Parkinson
//! 5. Realized Volatility (least efficient)
//!
//! # Examples
//!
//! ## Basic Realized Volatility
//! ```rust
//! use scirs2_series::financial::volatility::estimators::realized_volatility;
//! use scirs2_core::ndarray::array;
//!
//! let returns = array![0.01, -0.02, 0.015, -0.008, 0.012];
//! let realized_vol = realized_volatility(&returns);
//! ```
//!
//! ## Garman-Klass Estimator
//! ```rust
//! use scirs2_series::financial::volatility::estimators::garman_klass_volatility;
//! use scirs2_core::ndarray::array;
//!
//! let high = array![102.0, 105.0, 103.5];
//! let low = array![98.0, 101.0, 99.5];
//! let close = array![100.0, 103.0, 101.0];
//! let open = array![99.0, 102.0, 102.5];
//!
//! let gk_vol = garman_klass_volatility(&high, &low, &close, &open).unwrap();
//! ```
//!
//! ## EWMA Volatility
//! ```rust
//! use scirs2_series::financial::volatility::estimators::ewma_volatility;
//! use scirs2_core::ndarray::array;
//!
//! let returns = array![0.01, -0.02, 0.015, -0.008, 0.012];
//! let lambda = 0.94; // RiskMetrics standard
//! let ewma_vol = ewma_volatility(&returns, lambda).unwrap();
//! ```

use scirs2_core::ndarray::{s, Array1};
use scirs2_core::numeric::Float;

use crate::error::{Result, TimeSeriesError};

/// Calculate realized volatility from high-frequency returns
///
/// The simplest volatility estimator that sums squared returns.
/// This is the baseline estimator but least efficient as it only uses
/// closing prices.
///
/// # Formula
/// RV = Σ r²ᵢ where r is the return
///
/// # Arguments
///
/// * `returns` - Array of returns (not prices)
///
/// # Returns
///
/// * `F` - Realized volatility (single value)
///
/// # Examples
///
/// ```rust
/// use scirs2_series::financial::volatility::estimators::realized_volatility;
/// use scirs2_core::ndarray::array;
///
/// let returns = array![0.01, -0.02, 0.015, -0.008, 0.012];
/// let rv = realized_volatility(&returns);
/// ```
pub fn realized_volatility<F: Float>(returns: &Array1<F>) -> F {
    returns.mapv(|x| x * x).sum()
}

/// Garman-Klass volatility estimator
///
/// The most efficient unbiased volatility estimator using HLOC data.
/// It has 5 times lower variance than realized volatility for the same
/// sample period.
///
/// # Formula
/// GK = 0.5 * (ln(H/L))² - (2ln(2) - 1) * (ln(C/O))²
///
/// # Arguments
///
/// * `high` - High prices
/// * `low` - Low prices
/// * `close` - Closing prices
/// * `open` - Opening prices
///
/// # Returns
///
/// * `Result<Array1<F>>` - Garman-Klass volatility estimates
///
/// # Examples
///
/// ```rust
/// use scirs2_series::financial::volatility::estimators::garman_klass_volatility;
/// use scirs2_core::ndarray::array;
///
/// let high = array![102.0, 105.0, 103.5];
/// let low = array![98.0, 101.0, 99.5];
/// let close = array![100.0, 103.0, 101.0];
/// let open = array![99.0, 102.0, 102.5];
///
/// let gk_vol = garman_klass_volatility(&high, &low, &close, &open).unwrap();
/// ```
pub fn garman_klass_volatility<F: Float + Clone>(
    high: &Array1<F>,
    low: &Array1<F>,
    close: &Array1<F>,
    open: &Array1<F>,
) -> Result<Array1<F>> {
    if high.len() != low.len() || low.len() != close.len() || close.len() != open.len() {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: high.len(),
            actual: open.len(),
        });
    }

    let mut gk_vol = Array1::zeros(high.len());
    let half = F::from(0.5).unwrap();
    let ln_2_minus_1 = F::from(2.0 * (2.0_f64).ln() - 1.0).unwrap();

    for i in 0..gk_vol.len() {
        let log_hl = (high[i] / low[i]).ln();
        let log_co = (close[i] / open[i]).ln();

        gk_vol[i] = half * log_hl * log_hl - ln_2_minus_1 * log_co * log_co;
    }

    Ok(gk_vol)
}

/// Parkinson volatility estimator
///
/// Uses only high and low prices. Simple and robust but less efficient
/// than Garman-Klass. Good when opening prices are unreliable.
///
/// # Formula
/// P = (ln(H/L))² / (4 * ln(2))
///
/// # Arguments
///
/// * `high` - High prices
/// * `low` - Low prices
///
/// # Returns
///
/// * `Result<Array1<F>>` - Parkinson volatility estimates
///
/// # Examples
///
/// ```rust
/// use scirs2_series::financial::volatility::estimators::parkinson_volatility;
/// use scirs2_core::ndarray::array;
///
/// let high = array![102.0, 105.0, 103.5];
/// let low = array![98.0, 101.0, 99.5];
///
/// let park_vol = parkinson_volatility(&high, &low).unwrap();
/// ```
pub fn parkinson_volatility<F: Float + Clone>(
    high: &Array1<F>,
    low: &Array1<F>,
) -> Result<Array1<F>> {
    if high.len() != low.len() {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: high.len(),
            actual: low.len(),
        });
    }

    let mut park_vol = Array1::zeros(high.len());
    let four_ln_2 = F::from(4.0 * (2.0_f64).ln()).unwrap();

    for i in 0..park_vol.len() {
        let log_hl = (high[i] / low[i]).ln();
        park_vol[i] = log_hl * log_hl / four_ln_2;
    }

    Ok(park_vol)
}

/// Rogers-Satchell volatility estimator
///
/// A drift-independent estimator that uses HLOC data. Unlike Garman-Klass,
/// it's unaffected by drift in the underlying price process.
///
/// # Formula
/// RS = ln(H/O) * ln(C/O) + ln(L/O) * ln(C/O)
///
/// # Arguments
///
/// * `high` - High prices
/// * `low` - Low prices
/// * `close` - Closing prices
/// * `open` - Opening prices
///
/// # Returns
///
/// * `Result<Array1<F>>` - Rogers-Satchell volatility estimates
///
/// # Examples
///
/// ```rust
/// use scirs2_series::financial::volatility::estimators::rogers_satchell_volatility;
/// use scirs2_core::ndarray::array;
///
/// let high = array![102.0, 105.0, 103.5];
/// let low = array![98.0, 101.0, 99.5];
/// let close = array![100.0, 103.0, 101.0];
/// let open = array![99.0, 102.0, 102.5];
///
/// let rs_vol = rogers_satchell_volatility(&high, &low, &close, &open).unwrap();
/// ```
pub fn rogers_satchell_volatility<F: Float + Clone>(
    high: &Array1<F>,
    low: &Array1<F>,
    close: &Array1<F>,
    open: &Array1<F>,
) -> Result<Array1<F>> {
    if high.len() != low.len() || low.len() != close.len() || close.len() != open.len() {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: high.len(),
            actual: open.len(),
        });
    }

    let mut rs_vol = Array1::zeros(high.len());

    for i in 0..rs_vol.len() {
        let log_ho = (high[i] / open[i]).ln();
        let log_co = (close[i] / open[i]).ln();
        let log_lo = (low[i] / open[i]).ln();

        rs_vol[i] = log_ho * log_co + log_lo * log_co;
    }

    Ok(rs_vol)
}

/// Yang-Zhang volatility estimator
///
/// Combines overnight returns, open-to-close returns, and Rogers-Satchell
/// estimator. Handles opening gaps and is more robust to market microstructure
/// effects.
///
/// # Formula
/// YZ = σ²overnight + k * σ²open-close + σ²Rogers-Satchell
///
/// # Arguments
///
/// * `high` - High prices
/// * `low` - Low prices
/// * `close` - Closing prices
/// * `open` - Opening prices
/// * `k` - Weighting parameter for open-close component (typically 0.34)
///
/// # Returns
///
/// * `Result<Array1<F>>` - Yang-Zhang volatility estimates
///
/// # Examples
///
/// ```rust
/// use scirs2_series::financial::volatility::estimators::yang_zhang_volatility;
/// use scirs2_core::ndarray::array;
///
/// let high = array![102.0, 105.0, 103.5, 106.0];
/// let low = array![98.0, 101.0, 99.5, 102.0];
/// let close = array![100.0, 103.0, 101.0, 104.0];
/// let open = array![99.0, 102.0, 102.5, 100.5];
/// let k = 0.34;
///
/// let yz_vol = yang_zhang_volatility(&high, &low, &close, &open, k).unwrap();
/// ```
pub fn yang_zhang_volatility<F: Float + Clone>(
    high: &Array1<F>,
    low: &Array1<F>,
    close: &Array1<F>,
    open: &Array1<F>,
    k: F,
) -> Result<Array1<F>> {
    if high.len() != low.len() || low.len() != close.len() || close.len() != open.len() {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: high.len(),
            actual: open.len(),
        });
    }

    if high.len() < 2 {
        return Err(TimeSeriesError::InsufficientData {
            message: "Need at least 2 data points for Yang-Zhang volatility".to_string(),
            required: 2,
            actual: high.len(),
        });
    }

    let mut yz_vol = Array1::zeros(high.len() - 1);

    for i in 1..high.len() {
        // Overnight return
        let overnight = (open[i] / close[i - 1]).ln();

        // Open-to-close return
        let open_close = (close[i] / open[i]).ln();

        // Rogers-Satchell component
        let log_ho = (high[i] / open[i]).ln();
        let log_co = (close[i] / open[i]).ln();
        let log_lo = (low[i] / open[i]).ln();
        let rs = log_ho * log_co + log_lo * log_co;

        yz_vol[i - 1] = overnight * overnight + k * open_close * open_close + rs;
    }

    Ok(yz_vol)
}

/// GARCH(1,1) volatility estimation using simple method of moments
///
/// Implements a simplified GARCH(1,1) model for volatility estimation.
/// Uses rolling windows with typical parameter values for financial data.
///
/// # Model
/// σ²ₜ = ω + α * r²ₜ₋₁ + β * σ²ₜ₋₁
///
/// # Arguments
///
/// * `returns` - Return series
/// * `window` - Rolling window size for estimation
///
/// # Returns
///
/// * `Result<Array1<F>>` - GARCH volatility estimates
///
/// # Examples
///
/// ```rust
/// use scirs2_series::financial::volatility::estimators::garch_volatility_estimate;
/// use scirs2_core::ndarray::array;
///
/// let returns = array![0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.003, 0.007];
/// let garch_vol = garch_volatility_estimate(&returns, 5).unwrap();
/// ```
pub fn garch_volatility_estimate<F: Float + Clone>(
    returns: &Array1<F>,
    window: usize,
) -> Result<Array1<F>> {
    if returns.len() < window + 1 {
        return Err(TimeSeriesError::InsufficientData {
            message: "Not enough data for GARCH volatility estimation".to_string(),
            required: window + 1,
            actual: returns.len(),
        });
    }

    let mut volatilities = Array1::zeros(returns.len() - window + 1);

    // Simple GARCH(1,1) parameters (typical values)
    let omega = F::from(0.000001).unwrap();
    let alpha = F::from(0.1).unwrap();
    let beta = F::from(0.85).unwrap();

    for i in 0..volatilities.len() {
        let window_returns = returns.slice(s![i..i + window]);

        // Initialize with sample variance
        let mean = window_returns.sum() / F::from(window).unwrap();
        let mut variance =
            window_returns.mapv(|x| (x - mean).powi(2)).sum() / F::from(window - 1).unwrap();

        // Apply GARCH updating for last few observations
        for j in 1..std::cmp::min(window, 10) {
            let return_sq = window_returns[window - j].powi(2);
            variance = omega + alpha * return_sq + beta * variance;
        }

        volatilities[i] = variance.sqrt();
    }

    Ok(volatilities)
}

/// Exponentially Weighted Moving Average (EWMA) volatility
///
/// RiskMetrics-style EWMA volatility model. Uses exponential decay
/// to give more weight to recent observations.
///
/// # Model
/// σ²ₜ = λ * σ²ₜ₋₁ + (1-λ) * r²ₜ₋₁
///
/// # Arguments
///
/// * `returns` - Return series
/// * `lambda` - Decay parameter (RiskMetrics uses 0.94 for daily data)
///
/// # Returns
///
/// * `Result<Array1<F>>` - EWMA volatility estimates
///
/// # Examples
///
/// ```rust
/// use scirs2_series::financial::volatility::estimators::ewma_volatility;
/// use scirs2_core::ndarray::array;
///
/// let returns = array![0.01, -0.02, 0.015, -0.008, 0.012];
/// let lambda = 0.94; // RiskMetrics standard
/// let ewma_vol = ewma_volatility(&returns, lambda).unwrap();
/// ```
pub fn ewma_volatility<F: Float + Clone>(returns: &Array1<F>, lambda: F) -> Result<Array1<F>> {
    if returns.is_empty() {
        return Err(TimeSeriesError::InvalidInput(
            "Returns cannot be empty".to_string(),
        ));
    }

    if lambda <= F::zero() || lambda >= F::one() {
        return Err(TimeSeriesError::InvalidParameter {
            name: "lambda".to_string(),
            message: "Lambda must be between 0 and 1 (exclusive)".to_string(),
        });
    }

    let mut ewma_var = Array1::zeros(returns.len());

    // Initialize with first squared return
    ewma_var[0] = returns[0].powi(2);

    let one_minus_lambda = F::one() - lambda;

    for i in 1..returns.len() {
        ewma_var[i] = lambda * ewma_var[i - 1] + one_minus_lambda * returns[i].powi(2);
    }

    Ok(ewma_var.mapv(|x| x.sqrt()))
}

/// Range-based volatility using high-low range
///
/// A simple range-based estimator that uses the high-low range over
/// specified periods. Less efficient than specialized range estimators
/// but easy to compute.
///
/// # Formula
/// RV = sqrt(1/(4*ln(2)) * Σ(ln(H/L))² / n)
///
/// # Arguments
///
/// * `high` - High prices
/// * `low` - Low prices
/// * `period` - Rolling window period
///
/// # Returns
///
/// * `Result<Array1<F>>` - Range-based volatility estimates
///
/// # Examples
///
/// ```rust
/// use scirs2_series::financial::volatility::estimators::range_volatility;
/// use scirs2_core::ndarray::array;
///
/// let high = array![102.0, 105.0, 103.5, 106.0, 104.5];
/// let low = array![98.0, 101.0, 99.5, 102.0, 101.0];
/// let range_vol = range_volatility(&high, &low, 3).unwrap();
/// ```
pub fn range_volatility<F: Float + Clone>(
    high: &Array1<F>,
    low: &Array1<F>,
    period: usize,
) -> Result<Array1<F>> {
    if high.len() != low.len() {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: high.len(),
            actual: low.len(),
        });
    }

    if high.len() < period {
        return Err(TimeSeriesError::InsufficientData {
            message: "Not enough data for range volatility calculation".to_string(),
            required: period,
            actual: high.len(),
        });
    }

    let mut range_vol = Array1::zeros(high.len() - period + 1);
    let scaling_factor = F::from(1.0 / (4.0 * (2.0_f64).ln())).unwrap();

    for i in 0..range_vol.len() {
        let mut sum_log_range_sq = F::zero();

        for j in 0..period {
            let log_range = (high[i + j] / low[i + j]).ln();
            sum_log_range_sq = sum_log_range_sq + log_range.powi(2);
        }

        range_vol[i] = (scaling_factor * sum_log_range_sq / F::from(period).unwrap()).sqrt();
    }

    Ok(range_vol)
}

/// Intraday volatility estimation using tick data concept
///
/// Estimates volatility from high-frequency price observations by
/// calculating returns at specified sampling frequencies.
///
/// # Arguments
///
/// * `prices` - High-frequency price series
/// * `sampling_frequency` - Number of observations per sampling interval
///
/// # Returns
///
/// * `Result<F>` - Intraday volatility estimate (single value)
///
/// # Examples
///
/// ```rust
/// use scirs2_series::financial::volatility::estimators::intraday_volatility;
/// use scirs2_core::ndarray::array;
///
/// let prices = array![100.0, 100.1, 99.9, 100.05, 99.95, 100.2];
/// let sampling_freq = 2; // Every 2 observations
/// let intraday_vol = intraday_volatility(&prices, sampling_freq).unwrap();
/// ```
pub fn intraday_volatility<F: Float + Clone>(
    prices: &Array1<F>,
    sampling_frequency: usize,
) -> Result<F> {
    if prices.len() < sampling_frequency + 1 {
        return Err(TimeSeriesError::InsufficientData {
            message: "Not enough data for intraday volatility calculation".to_string(),
            required: sampling_frequency + 1,
            actual: prices.len(),
        });
    }

    let mut squared_returns = F::zero();
    let mut count = 0;

    for i in sampling_frequency..prices.len() {
        let logreturn = (prices[i] / prices[i - sampling_frequency]).ln();
        squared_returns = squared_returns + logreturn.powi(2);
        count += 1;
    }

    if count == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "No valid returns calculated".to_string(),
        ));
    }

    Ok((squared_returns / F::from(count).unwrap()).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr1;

    #[test]
    fn test_realized_volatility() {
        let returns = arr1(&[0.01, -0.02, 0.015, -0.008, 0.012]);
        let rv = realized_volatility(&returns);

        // Should equal sum of squared returns
        let expected: f64 = 0.01_f64.powi(2)
            + 0.02_f64.powi(2)
            + 0.015_f64.powi(2)
            + 0.008_f64.powi(2)
            + 0.012_f64.powi(2);
        assert!((rv - expected).abs() < 1e-10);
    }

    #[test]
    fn test_garman_klass_volatility() {
        let high = arr1(&[102.0, 105.0, 103.5]);
        let low = arr1(&[98.0, 101.0, 99.5]);
        let close = arr1(&[100.0, 103.0, 101.0]);
        let open = arr1(&[99.0, 102.0, 102.5]);

        let result = garman_klass_volatility(&high, &low, &close, &open);
        assert!(result.is_ok());

        let gk_vol = result.unwrap();
        assert_eq!(gk_vol.len(), 3);

        // All values should be non-negative
        for &vol in gk_vol.iter() {
            assert!(vol >= 0.0);
        }
    }

    #[test]
    fn test_parkinson_volatility() {
        let high = arr1(&[102.0, 105.0, 103.5]);
        let low = arr1(&[98.0, 101.0, 99.5]);

        let result = parkinson_volatility(&high, &low);
        assert!(result.is_ok());

        let park_vol = result.unwrap();
        assert_eq!(park_vol.len(), 3);

        // All values should be non-negative
        for &vol in park_vol.iter() {
            assert!(vol >= 0.0);
        }
    }

    #[test]
    fn test_rogers_satchell_volatility() {
        let high = arr1(&[102.0, 105.0, 103.5]);
        let low = arr1(&[98.0, 101.0, 99.5]);
        let close = arr1(&[100.0, 103.0, 101.0]);
        let open = arr1(&[99.0, 102.0, 102.5]);

        let result = rogers_satchell_volatility(&high, &low, &close, &open);
        assert!(result.is_ok());

        let rs_vol = result.unwrap();
        assert_eq!(rs_vol.len(), 3);
    }

    #[test]
    fn test_yang_zhang_volatility() {
        let high = arr1(&[102.0, 105.0, 103.5, 106.0]);
        let low = arr1(&[98.0, 101.0, 99.5, 102.0]);
        let close = arr1(&[100.0, 103.0, 101.0, 104.0]);
        let open = arr1(&[99.0, 102.0, 102.5, 100.5]);
        let k = 0.34;

        let result = yang_zhang_volatility(&high, &low, &close, &open, k);
        assert!(result.is_ok());

        let yz_vol = result.unwrap();
        assert_eq!(yz_vol.len(), 3); // n-1 for n observations
    }

    #[test]
    fn test_ewma_volatility() {
        let returns = arr1(&[0.01, -0.02, 0.015, -0.008, 0.012]);
        let lambda = 0.94;

        let result = ewma_volatility(&returns, lambda);
        assert!(result.is_ok());

        let ewma_vol = result.unwrap();
        assert_eq!(ewma_vol.len(), returns.len());

        // First value should be sqrt of first squared return
        assert!((ewma_vol[0] - (returns[0] * returns[0]).sqrt()).abs() < 1e-10);

        // All values should be positive
        for &vol in ewma_vol.iter() {
            assert!(vol >= 0.0);
        }
    }

    #[test]
    fn test_garch_volatility_estimate() {
        let returns = arr1(&[0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.003, 0.007]);
        let window = 5;

        let result = garch_volatility_estimate(&returns, window);
        assert!(result.is_ok());

        let garch_vol = result.unwrap();
        assert_eq!(garch_vol.len(), returns.len() - window + 1);

        // All values should be positive
        for &vol in garch_vol.iter() {
            assert!(vol > 0.0);
        }
    }

    #[test]
    fn test_range_volatility() {
        let high = arr1(&[102.0, 105.0, 103.5, 106.0, 104.5]);
        let low = arr1(&[98.0, 101.0, 99.5, 102.0, 101.0]);
        let period = 3;

        let result = range_volatility(&high, &low, period);
        assert!(result.is_ok());

        let range_vol = result.unwrap();
        assert_eq!(range_vol.len(), high.len() - period + 1);

        // All values should be non-negative
        for &vol in range_vol.iter() {
            assert!(vol >= 0.0);
        }
    }

    #[test]
    fn test_intraday_volatility() {
        let prices = arr1(&[100.0, 100.1, 99.9, 100.05, 99.95, 100.2]);
        let sampling_freq = 2;

        let result = intraday_volatility(&prices, sampling_freq);
        assert!(result.is_ok());

        let vol = result.unwrap();
        assert!(vol >= 0.0);
    }

    #[test]
    fn test_dimension_mismatch() {
        let high = arr1(&[102.0, 105.0]);
        let low = arr1(&[98.0, 101.0, 99.5]);

        let result = parkinson_volatility(&high, &low);
        assert!(result.is_err());
    }

    #[test]
    fn test_insufficient_data() {
        let returns = arr1(&[0.01]);
        let window = 5;

        let result = garch_volatility_estimate(&returns, window);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_parameters() {
        let returns = arr1(&[0.01, -0.02, 0.015]);

        // Invalid lambda > 1
        let result = ewma_volatility(&returns, 1.1);
        assert!(result.is_err());

        // Invalid lambda <= 0
        let result = ewma_volatility(&returns, 0.0);
        assert!(result.is_err());
    }
}
