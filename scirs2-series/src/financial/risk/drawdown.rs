//! Drawdown analysis for portfolio and asset performance
//!
//! This module provides comprehensive drawdown analysis tools for measuring
//! the peak-to-trough decline during a specific period. Drawdown analysis
//! is crucial for understanding the downside risk and recovery characteristics
//! of investment strategies.
//!
//! # Overview
//!
//! Drawdown metrics help investors and portfolio managers:
//! - Assess maximum loss potential from peak levels
//! - Understand recovery time after losses
//! - Evaluate risk-adjusted performance
//! - Set position sizing and risk limits
//!
//! # Key Metrics
//!
//! ## Maximum Drawdown (MDD)
//! The largest peak-to-trough decline, expressed as a percentage.
//! This is the most commonly used drawdown metric.
//!
//! ## Pain Index
//! The average drawdown over time, providing a measure of sustained
//! loss periods rather than just peak losses.
//!
//! ## Ulcer Index
//! The root mean square (RMS) of all drawdowns, giving higher weight
//! to larger drawdowns while considering their duration.
//!
//! ## Calmar Ratio
//! Annual return divided by maximum drawdown, showing return per
//! unit of worst-case loss.
//!
//! ## Recovery Analysis
//! Time-based analysis of how long it takes to recover from drawdowns.
//!
//! # Examples
//!
//! ## Basic Drawdown Analysis
//! ```rust
//! use scirs2_series::financial::risk::drawdown::{max_drawdown, calculate_drawdown_series};
//! use scirs2_core::ndarray::array;
//!
//! // Portfolio value over time
//! let portfolio_values = array![1000.0, 1100.0, 1050.0, 950.0, 1200.0, 1150.0, 1300.0];
//!
//! // Maximum drawdown
//! let mdd = max_drawdown(&portfolio_values).unwrap();
//! println!("Maximum Drawdown: {:.2}%", mdd * 100.0);
//!
//! // Full drawdown series
//! let drawdowns = calculate_drawdown_series(&portfolio_values).unwrap();
//! println!("Drawdown series: {:?}", drawdowns);
//! ```
//!
//! ## Advanced Drawdown Metrics
//! ```rust
//! use scirs2_series::financial::risk::drawdown::{pain_index, ulcer_index, calmar_ratio};
//! use scirs2_core::ndarray::array;
//!
//! let portfolio_values = array![1000.0, 1100.0, 1050.0, 950.0, 1200.0, 1150.0, 1300.0];
//! let returns = array![0.10, -0.05, -0.10, 0.26, -0.04, 0.13]; // Corresponding returns
//!
//! // Pain Index - average drawdown
//! let pain = pain_index(&portfolio_values).unwrap();
//!
//! // Ulcer Index - RMS drawdown
//! let ulcer = ulcer_index(&portfolio_values).unwrap();
//!
//! // Calmar Ratio - return/max drawdown trade-off
//! let calmar = calmar_ratio(&returns, &portfolio_values, 252).unwrap();
//! ```
//!
//! ## Drawdown Recovery Analysis
//! ```rust
//! use scirs2_series::financial::risk::drawdown::{drawdown_recovery_analysis, DrawdownPeriod};
//! use scirs2_core::ndarray::array;
//!
//! let portfolio_values = array![1000.0, 1100.0, 1050.0, 950.0, 1200.0, 1150.0, 1300.0];
//!
//! let recovery_analysis = drawdown_recovery_analysis(&portfolio_values).unwrap();
//! for period in recovery_analysis {
//!     println!("Drawdown: {:.2}%, Duration: {} periods, Recovery: {} periods",
//!              period.max_drawdown * 100.0, period.duration,
//!              period.recovery_periods.unwrap_or(0));
//! }
//! ```

use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;

use crate::error::{Result, TimeSeriesError};

/// Drawdown period information
#[derive(Debug, Clone)]
pub struct DrawdownPeriod<F: Float> {
    /// Start index of drawdown period
    pub start_index: usize,
    /// End index of drawdown period (trough)
    pub end_index: usize,
    /// Recovery index (when portfolio recovers to peak), None if not recovered
    pub recovery_index: Option<usize>,
    /// Maximum drawdown during this period
    pub max_drawdown: F,
    /// Duration of drawdown (start to trough)
    pub duration: usize,
    /// Recovery periods (trough to recovery), None if not recovered
    pub recovery_periods: Option<usize>,
    /// Peak value at start of drawdown
    pub peak_value: F,
    /// Trough value
    pub trough_value: F,
}

/// Calculate maximum drawdown from portfolio values
///
/// Maximum drawdown is the largest peak-to-trough decline expressed as
/// a percentage. It represents the worst-case loss an investor would
/// have experienced during the period.
///
/// # Arguments
///
/// * `values` - Portfolio or asset values over time
///
/// # Returns
///
/// * `Result<F>` - Maximum drawdown as a decimal (0.1 = 10% drawdown)
///
/// # Examples
///
/// ```rust
/// use scirs2_series::financial::risk::drawdown::max_drawdown;
/// use scirs2_core::ndarray::array;
///
/// let portfolio_values = array![1000.0, 1100.0, 1050.0, 950.0, 1200.0];
/// let mdd = max_drawdown(&portfolio_values).unwrap();
/// println!("Max drawdown: {:.2}%", mdd * 100.0);
/// ```
pub fn max_drawdown<F: Float + Clone>(values: &Array1<F>) -> Result<F> {
    if values.is_empty() {
        return Err(TimeSeriesError::InvalidInput(
            "Values cannot be empty".to_string(),
        ));
    }

    let mut max_value = values[0];
    let mut max_dd = F::zero();

    for &value in values.iter() {
        if value > max_value {
            max_value = value;
        }

        let drawdown = (max_value - value) / max_value;
        if drawdown > max_dd {
            max_dd = drawdown;
        }
    }

    Ok(max_dd)
}

/// Calculate drawdown series from portfolio values
///
/// Returns the drawdown at each point in time, showing the running
/// peak-to-current decline as a percentage.
///
/// # Arguments
///
/// * `values` - Portfolio or asset values over time
///
/// # Returns
///
/// * `Result<Array1<F>>` - Drawdown series (negative values indicate drawdowns)
///
/// # Examples
///
/// ```rust
/// use scirs2_series::financial::risk::drawdown::calculate_drawdown_series;
/// use scirs2_core::ndarray::array;
///
/// let values = array![1000.0, 1100.0, 1050.0, 950.0, 1200.0];
/// let drawdowns = calculate_drawdown_series(&values).unwrap();
/// ```
pub fn calculate_drawdown_series<F: Float + Clone>(values: &Array1<F>) -> Result<Array1<F>> {
    if values.is_empty() {
        return Err(TimeSeriesError::InvalidInput(
            "Values cannot be empty".to_string(),
        ));
    }

    let mut drawdowns = Array1::zeros(values.len());
    let mut peak = values[0];

    for i in 0..values.len() {
        if values[i] > peak {
            peak = values[i];
        }
        drawdowns[i] = (values[i] - peak) / peak;
    }

    Ok(drawdowns)
}

/// Calculate Pain Index (average drawdown)
///
/// The Pain Index measures the average drawdown over the entire period.
/// It provides insight into sustained loss periods rather than just
/// peak losses.
///
/// # Arguments
///
/// * `values` - Portfolio or asset values over time
///
/// # Returns
///
/// * `Result<F>` - Pain Index as a decimal
///
/// # Examples
///
/// ```rust
/// use scirs2_series::financial::risk::drawdown::pain_index;
/// use scirs2_core::ndarray::array;
///
/// let values = array![1000.0, 1100.0, 1050.0, 950.0, 1200.0];
/// let pain = pain_index(&values).unwrap();
/// ```
pub fn pain_index<F: Float + Clone>(values: &Array1<F>) -> Result<F> {
    let drawdowns = calculate_drawdown_series(values)?;

    // Convert to positive values and average
    let total_drawdown = drawdowns.mapv(|x| -x).sum();
    Ok(total_drawdown / F::from(values.len()).unwrap())
}

/// Calculate Ulcer Index (RMS of drawdowns)
///
/// The Ulcer Index is the root mean square of all drawdowns, giving
/// higher weight to larger drawdowns while considering their duration.
/// It's named after the "ulcer" that large drawdowns can cause investors.
///
/// # Arguments
///
/// * `values` - Portfolio or asset values over time
///
/// # Returns
///
/// * `Result<F>` - Ulcer Index as a decimal
///
/// # Examples
///
/// ```rust
/// use scirs2_series::financial::risk::drawdown::ulcer_index;
/// use scirs2_core::ndarray::array;
///
/// let values = array![1000.0, 1100.0, 1050.0, 950.0, 1200.0];
/// let ulcer = ulcer_index(&values).unwrap();
/// ```
pub fn ulcer_index<F: Float + Clone>(values: &Array1<F>) -> Result<F> {
    let drawdowns = calculate_drawdown_series(values)?;

    // Calculate RMS of drawdowns (as positive values)
    let sum_squared_dd = drawdowns.mapv(|x| x.powi(2)).sum();
    Ok((sum_squared_dd / F::from(values.len()).unwrap()).sqrt())
}

/// Calculate Calmar Ratio (annual return / maximum drawdown)
///
/// The Calmar Ratio measures the trade-off between return and maximum drawdown.
/// Higher values indicate better risk-adjusted performance relative to
/// worst-case scenarios.
///
/// # Arguments
///
/// * `returns` - Return series
/// * `values` - Portfolio values for drawdown calculation
/// * `periods_per_year` - Number of periods per year for annualization
///
/// # Returns
///
/// * `Result<F>` - Calmar Ratio
///
/// # Examples
///
/// ```rust
/// use scirs2_series::financial::risk::drawdown::calmar_ratio;
/// use scirs2_core::ndarray::array;
///
/// let returns = array![0.10, -0.05, -0.10, 0.26, -0.04, 0.13];
/// let values = array![1000.0, 1100.0, 1050.0, 950.0, 1200.0, 1150.0, 1300.0];
/// let calmar = calmar_ratio(&returns, &values, 252).unwrap();
/// ```
pub fn calmar_ratio<F: Float + Clone>(
    returns: &Array1<F>,
    values: &Array1<F>,
    periods_per_year: usize,
) -> Result<F> {
    if returns.is_empty() || values.is_empty() {
        return Err(TimeSeriesError::InvalidInput(
            "Returns and values cannot be empty".to_string(),
        ));
    }

    // Calculate annualized return
    let total_return = (values[values.len() - 1] / values[0]) - F::one();
    let years = F::from(returns.len()).unwrap() / F::from(periods_per_year).unwrap();
    let annualized_return = (F::one() + total_return).powf(F::one() / years) - F::one();

    // Calculate maximum drawdown
    let mdd = max_drawdown(values)?;

    if mdd == F::zero() {
        Ok(F::infinity())
    } else {
        Ok(annualized_return / mdd)
    }
}

/// Perform comprehensive drawdown recovery analysis
///
/// Identifies all drawdown periods and their recovery characteristics,
/// providing detailed information about each significant drawdown event.
///
/// # Arguments
///
/// * `values` - Portfolio or asset values over time
///
/// # Returns
///
/// * `Result<Vec<DrawdownPeriod<F>>>` - Vector of drawdown periods with recovery analysis
///
/// # Examples
///
/// ```rust
/// use scirs2_series::financial::risk::drawdown::drawdown_recovery_analysis;
/// use scirs2_core::ndarray::array;
///
/// let values = array![1000.0, 1100.0, 1050.0, 950.0, 1200.0, 1150.0, 1300.0];
/// let analysis = drawdown_recovery_analysis(&values).unwrap();
/// ```
pub fn drawdown_recovery_analysis<F: Float + Clone>(
    values: &Array1<F>,
) -> Result<Vec<DrawdownPeriod<F>>> {
    if values.is_empty() {
        return Err(TimeSeriesError::InvalidInput(
            "Values cannot be empty".to_string(),
        ));
    }

    let mut periods = Vec::new();
    let mut peak = values[0];
    let mut peak_index = 0;
    let mut in_drawdown = false;
    let mut trough_value = values[0];
    let mut trough_index = 0;

    for i in 0..values.len() {
        if values[i] > peak {
            // New peak
            if in_drawdown {
                // End of drawdown period, look for recovery
                let recovery_index = find_recovery_index(values, i, peak);
                let duration = trough_index - peak_index;
                let recovery_periods = recovery_index.map(|idx| idx - trough_index);
                let max_dd = (peak - trough_value) / peak;

                periods.push(DrawdownPeriod {
                    start_index: peak_index,
                    end_index: trough_index,
                    recovery_index,
                    max_drawdown: max_dd,
                    duration,
                    recovery_periods,
                    peak_value: peak,
                    trough_value,
                });

                in_drawdown = false;
            }

            peak = values[i];
            peak_index = i;
        } else if values[i] < peak {
            // In drawdown
            if !in_drawdown {
                in_drawdown = true;
                trough_value = values[i];
                trough_index = i;
            } else if values[i] < trough_value {
                // New trough
                trough_value = values[i];
                trough_index = i;
            }
        }
    }

    // Handle case where we end in drawdown
    if in_drawdown {
        let duration = trough_index - peak_index;
        let max_dd = (peak - trough_value) / peak;

        periods.push(DrawdownPeriod {
            start_index: peak_index,
            end_index: trough_index,
            recovery_index: None,
            max_drawdown: max_dd,
            duration,
            recovery_periods: None,
            peak_value: peak,
            trough_value,
        });
    }

    Ok(periods)
}

/// Calculate maximum consecutive losses count
///
/// Counts the maximum number of consecutive negative returns,
/// which helps understand losing streaks.
///
/// # Arguments
///
/// * `returns` - Return series
///
/// # Returns
///
/// * `usize` - Maximum consecutive losses
pub fn max_consecutive_losses<F: Float + Clone>(returns: &Array1<F>) -> usize {
    let mut max_consecutive = 0;
    let mut current_consecutive = 0;

    for &ret in returns.iter() {
        if ret < F::zero() {
            current_consecutive += 1;
            max_consecutive = max_consecutive.max(current_consecutive);
        } else {
            current_consecutive = 0;
        }
    }

    max_consecutive
}

/// Calculate average drawdown duration
///
/// Measures the average time it takes from peak to trough across
/// all drawdown periods.
///
/// # Arguments
///
/// * `values` - Portfolio or asset values over time
///
/// # Returns
///
/// * `Result<F>` - Average drawdown duration in periods
pub fn average_drawdown_duration<F: Float + Clone>(values: &Array1<F>) -> Result<F> {
    let periods = drawdown_recovery_analysis(values)?;

    if periods.is_empty() {
        return Ok(F::zero());
    }

    let total_duration: usize = periods.iter().map(|p| p.duration).sum();
    Ok(F::from(total_duration).unwrap() / F::from(periods.len()).unwrap())
}

/// Calculate average recovery time
///
/// Measures the average time it takes to recover from trough back to peak
/// across all recovered drawdown periods.
///
/// # Arguments
///
/// * `values` - Portfolio or asset values over time
///
/// # Returns
///
/// * `Result<F>` - Average recovery time in periods (only for recovered drawdowns)
pub fn average_recovery_time<F: Float + Clone>(values: &Array1<F>) -> Result<F> {
    let periods = drawdown_recovery_analysis(values)?;

    let recovered_periods: Vec<&DrawdownPeriod<F>> = periods
        .iter()
        .filter(|p| p.recovery_periods.is_some())
        .collect();

    if recovered_periods.is_empty() {
        return Ok(F::zero());
    }

    let total_recovery: usize = recovered_periods
        .iter()
        .map(|p| p.recovery_periods.unwrap())
        .sum();

    Ok(F::from(total_recovery).unwrap() / F::from(recovered_periods.len()).unwrap())
}

/// Helper function to find recovery index
fn find_recovery_index<F: Float + Clone>(
    values: &Array1<F>,
    start_search: usize,
    target_peak: F,
) -> Option<usize> {
    (start_search..values.len()).find(|&i| values[i] >= target_peak)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr1;

    #[test]
    fn test_max_drawdown() {
        let values = arr1(&[1000.0, 1100.0, 1050.0, 950.0, 1200.0, 1150.0, 1300.0]);
        let mdd = max_drawdown(&values).unwrap();

        // Maximum drawdown should be from peak of 1100 to trough of 950
        let expected = (1100.0 - 950.0) / 1100.0;
        assert!((mdd - expected).abs() < 1e-10);
    }

    #[test]
    fn test_drawdown_series() {
        let values = arr1(&[1000.0, 1100.0, 1050.0, 950.0, 1200.0]);
        let drawdowns = calculate_drawdown_series(&values).unwrap();

        assert_eq!(drawdowns.len(), values.len());

        // First value should have zero drawdown
        assert!(drawdowns[0] == 0.0);

        // Drawdowns should be non-positive
        for &dd in drawdowns.iter() {
            assert!(dd <= 0.0);
        }
    }

    #[test]
    fn test_pain_index() {
        let values = arr1(&[1000.0, 1100.0, 1050.0, 950.0, 1200.0]);
        let pain = pain_index(&values).unwrap();

        // Pain index should be positive (we convert drawdowns to positive)
        assert!(pain >= 0.0);
    }

    #[test]
    fn test_ulcer_index() {
        let values = arr1(&[1000.0, 1100.0, 1050.0, 950.0, 1200.0]);
        let ulcer = ulcer_index(&values).unwrap();

        // Ulcer index should be positive
        assert!(ulcer >= 0.0);
    }

    #[test]
    fn test_calmar_ratio() {
        let returns = arr1(&[0.10, -0.05, -0.10, 0.26, -0.04]);
        let values = arr1(&[1000.0, 1100.0, 1050.0, 950.0, 1200.0, 1150.0]);

        let result = calmar_ratio(&returns, &values, 252);
        assert!(result.is_ok());

        let calmar = result.unwrap();
        // Calmar ratio should be finite or infinity (if no drawdown)
        assert!(calmar.is_finite() || calmar.is_infinite());
    }

    #[test]
    fn test_drawdown_recovery_analysis() {
        let values = arr1(&[1000.0, 1100.0, 1050.0, 950.0, 1200.0, 1150.0, 1300.0]);
        let periods = drawdown_recovery_analysis(&values).unwrap();

        // Should have at least one drawdown period
        assert!(!periods.is_empty());

        // Check properties of first period
        if !periods.is_empty() {
            let first_period = &periods[0];
            assert!(first_period.max_drawdown > 0.0);
            assert!(first_period.duration > 0);
            assert!(first_period.peak_value >= first_period.trough_value);
        }
    }

    #[test]
    fn test_max_consecutive_losses() {
        let returns = arr1(&[0.01, -0.02, -0.01, -0.005, 0.02, -0.01, 0.03]);
        let max_losses = max_consecutive_losses(&returns);

        // Should be 3 consecutive losses in the middle
        assert_eq!(max_losses, 3);
    }

    #[test]
    fn test_no_drawdown() {
        let values = arr1(&[1000.0, 1100.0, 1200.0, 1300.0, 1400.0]);
        let mdd = max_drawdown(&values).unwrap();

        // Should be zero drawdown for monotonically increasing series
        assert!(mdd == 0.0);
    }

    #[test]
    fn test_empty_input() {
        let values: Array1<f64> = arr1(&[]);
        let result = max_drawdown(&values);
        assert!(result.is_err());
    }

    #[test]
    fn test_single_value() {
        let values = arr1(&[1000.0]);
        let mdd = max_drawdown(&values).unwrap();

        // Single value should have zero drawdown
        assert!(mdd == 0.0);
    }
}
