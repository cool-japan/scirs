//! Portfolio performance metrics and calculation functions
//!
//! This module provides comprehensive portfolio performance analysis tools,
//! including risk-adjusted returns, volatility measures, and various
//! performance ratios used in quantitative portfolio management.

use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;

use super::core::PortfolioMetrics;
use crate::error::{Result, TimeSeriesError};

/// Calculate comprehensive portfolio metrics
///
/// Computes a full suite of portfolio performance metrics including returns,
/// risk measures, and risk-adjusted performance ratios.
///
/// # Arguments
///
/// * `returns` - Portfolio return time series
/// * `prices` - Portfolio value time series for drawdown calculation
/// * `risk_free_rate` - Annual risk-free rate for ratio calculations
/// * `periods_per_year` - Number of periods per year for annualization
///
/// # Returns
///
/// * `Result<PortfolioMetrics<F>>` - Complete performance metrics structure
///
/// # Examples
///
/// ```rust
/// use scirs2_series::financial::portfolio::metrics::calculate_portfolio_metrics;
/// use scirs2_core::ndarray::array;
///
/// let returns = array![0.01, -0.02, 0.015, -0.008, 0.012];
/// let prices = array![1000.0, 1010.0, 989.8, 1004.64, 996.6, 1008.6];
/// let risk_free_rate = 0.02;
/// let periods_per_year = 252;
///
/// let metrics = calculate_portfolio_metrics(&returns, &prices, risk_free_rate, periods_per_year).unwrap();
/// ```
pub fn calculate_portfolio_metrics<F: Float + Clone + std::iter::Sum>(
    returns: &Array1<F>,
    prices: &Array1<F>,
    risk_free_rate: F,
    periods_per_year: usize,
) -> Result<PortfolioMetrics<F>> {
    if returns.is_empty() || prices.is_empty() {
        return Err(TimeSeriesError::InvalidInput(
            "Returns and prices cannot be empty".to_string(),
        ));
    }

    if prices.len() != returns.len() + 1 {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: returns.len() + 1,
            actual: prices.len(),
        });
    }

    // Total return
    let total_return = (prices[prices.len() - 1] / prices[0]) - F::one();

    // Annualized return
    let years = F::from(returns.len()).unwrap() / F::from(periods_per_year).unwrap();
    let annualized_return = (F::one() + total_return).powf(F::one() / years) - F::one();

    // Volatility (annualized)
    let mean_return = returns.sum() / F::from(returns.len()).unwrap();
    let variance =
        returns.mapv(|r| (r - mean_return).powi(2)).sum() / F::from(returns.len() - 1).unwrap();
    let volatility = variance.sqrt() * F::from(periods_per_year).unwrap().sqrt();

    // Risk metrics using imported functions
    let sharpe = crate::financial::risk::sharpe_ratio(returns, risk_free_rate, periods_per_year)?;
    let sortino = crate::financial::risk::sortino_ratio(returns, risk_free_rate, periods_per_year)?;
    let max_dd = crate::financial::risk::max_drawdown(prices)?;
    let calmar = crate::financial::risk::calmar_ratio(returns, prices, periods_per_year)?;
    let var_95 = crate::financial::risk::var_historical(returns, 0.95)?;
    let es_95 = crate::financial::risk::expected_shortfall(returns, 0.95)?;

    Ok(PortfolioMetrics::new(
        total_return,
        annualized_return,
        volatility,
        sharpe,
        sortino,
        max_dd,
        calmar,
        var_95,
        es_95,
    ))
}

/// Calculate portfolio return statistics
///
/// Computes basic return statistics including mean, standard deviation,
/// skewness, and kurtosis of the return distribution.
///
/// # Arguments
///
/// * `returns` - Portfolio return time series
///
/// # Returns
///
/// * `Result<(F, F, F, F)>` - (mean, std_dev, skewness, kurtosis)
pub fn calculate_return_statistics<F: Float + Clone>(returns: &Array1<F>) -> Result<(F, F, F, F)> {
    if returns.len() < 4 {
        return Err(TimeSeriesError::InsufficientData {
            message: "Need at least 4 observations for return statistics".to_string(),
            required: 4,
            actual: returns.len(),
        });
    }

    let n = F::from(returns.len()).unwrap();
    let mean = returns.sum() / n;

    // Calculate centered moments
    let mut second_moment = F::zero();
    let mut third_moment = F::zero();
    let mut fourth_moment = F::zero();

    for &ret in returns.iter() {
        let deviation = ret - mean;
        let dev_squared = deviation.powi(2);
        let dev_cubed = deviation.powi(3);
        let dev_fourth = deviation.powi(4);

        second_moment = second_moment + dev_squared;
        third_moment = third_moment + dev_cubed;
        fourth_moment = fourth_moment + dev_fourth;
    }

    second_moment = second_moment / (n - F::one());
    third_moment = third_moment / n;
    fourth_moment = fourth_moment / n;

    let std_dev = second_moment.sqrt();
    let variance = second_moment;

    // Skewness
    let skewness = if variance > F::zero() {
        third_moment / variance.powf(F::from(1.5).unwrap())
    } else {
        F::zero()
    };

    // Excess kurtosis
    let kurtosis = if variance > F::zero() {
        (fourth_moment / variance.powi(2)) - F::from(3.0).unwrap()
    } else {
        F::zero()
    };

    Ok((mean, std_dev, skewness, kurtosis))
}

/// Calculate rolling performance metrics
///
/// Computes performance metrics over rolling windows to analyze
/// time-varying portfolio characteristics.
///
/// # Arguments
///
/// * `returns` - Portfolio return time series
/// * `window` - Rolling window size
/// * `risk_free_rate` - Annual risk-free rate
/// * `periods_per_year` - Number of periods per year
///
/// # Returns
///
/// * `Result<(Array1<F>, Array1<F>, Array1<F>)>` - (rolling_sharpe, rolling_volatility, rolling_returns)
pub fn calculate_rolling_metrics<F: Float + Clone>(
    returns: &Array1<F>,
    window: usize,
    risk_free_rate: F,
    periods_per_year: usize,
) -> Result<(Array1<F>, Array1<F>, Array1<F>)> {
    if returns.len() < window {
        return Err(TimeSeriesError::InsufficientData {
            message: "Insufficient data for rolling window".to_string(),
            required: window,
            actual: returns.len(),
        });
    }

    let n_windows = returns.len() - window + 1;
    let mut rolling_sharpe = Array1::zeros(n_windows);
    let mut rolling_volatility = Array1::zeros(n_windows);
    let mut rolling_returns = Array1::zeros(n_windows);

    let annualized_rf = risk_free_rate / F::from(periods_per_year).unwrap();

    for i in 0..n_windows {
        let window_returns = returns.slice(scirs2_core::ndarray::s![i..i + window]);

        // Calculate window metrics
        let mean_return = window_returns.sum() / F::from(window).unwrap();
        let excess_return = mean_return - annualized_rf;

        let variance =
            window_returns.mapv(|r| (r - mean_return).powi(2)).sum() / F::from(window - 1).unwrap();
        let std_dev = variance.sqrt();

        // Annualize metrics
        rolling_returns[i] = mean_return * F::from(periods_per_year).unwrap();
        rolling_volatility[i] = std_dev * F::from(periods_per_year).unwrap().sqrt();

        rolling_sharpe[i] = if std_dev > F::zero() {
            (excess_return * F::from(periods_per_year).unwrap()) / rolling_volatility[i]
        } else {
            F::zero()
        };
    }

    Ok((rolling_sharpe, rolling_volatility, rolling_returns))
}

/// Calculate tracking error relative to benchmark
///
/// Measures the standard deviation of active returns (portfolio - benchmark),
/// indicating how closely the portfolio tracks a benchmark.
///
/// # Arguments
///
/// * `portfolio_returns` - Portfolio return series
/// * `benchmark_returns` - Benchmark return series
///
/// # Returns
///
/// * `Result<F>` - Annualized tracking error
pub fn calculate_tracking_error<F: Float + Clone>(
    portfolio_returns: &Array1<F>,
    benchmark_returns: &Array1<F>,
) -> Result<F> {
    if portfolio_returns.len() != benchmark_returns.len() {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: portfolio_returns.len(),
            actual: benchmark_returns.len(),
        });
    }

    if portfolio_returns.len() < 2 {
        return Err(TimeSeriesError::InsufficientData {
            message: "Need at least 2 observations for tracking error".to_string(),
            required: 2,
            actual: portfolio_returns.len(),
        });
    }

    // Calculate active returns
    let active_returns: Array1<F> = portfolio_returns
        .iter()
        .zip(benchmark_returns.iter())
        .map(|(&p, &b)| p - b)
        .collect();

    // Calculate tracking error (standard deviation of active returns)
    let mean_active = active_returns.sum() / F::from(active_returns.len()).unwrap();
    let variance = active_returns.mapv(|r| (r - mean_active).powi(2)).sum()
        / F::from(active_returns.len() - 1).unwrap();

    Ok(variance.sqrt())
}

/// Calculate information ratio
///
/// Measures risk-adjusted active return by dividing mean active return
/// by tracking error. Higher values indicate better active management.
///
/// # Arguments
///
/// * `portfolio_returns` - Portfolio return series
/// * `benchmark_returns` - Benchmark return series
///
/// # Returns
///
/// * `Result<F>` - Information ratio
pub fn calculate_information_ratio<F: Float + Clone>(
    portfolio_returns: &Array1<F>,
    benchmark_returns: &Array1<F>,
) -> Result<F> {
    if portfolio_returns.len() != benchmark_returns.len() {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: portfolio_returns.len(),
            actual: benchmark_returns.len(),
        });
    }

    let active_returns: Array1<F> = portfolio_returns
        .iter()
        .zip(benchmark_returns.iter())
        .map(|(&p, &b)| p - b)
        .collect();

    let mean_active = active_returns.sum() / F::from(active_returns.len()).unwrap();
    let tracking_error = calculate_tracking_error(portfolio_returns, benchmark_returns)?;

    if tracking_error == F::zero() {
        Ok(F::infinity())
    } else {
        Ok(mean_active / tracking_error)
    }
}

/// Calculate portfolio beta relative to benchmark
///
/// Measures systematic risk by calculating the covariance between
/// portfolio and benchmark returns divided by benchmark variance.
///
/// # Arguments
///
/// * `portfolio_returns` - Portfolio return series
/// * `benchmark_returns` - Benchmark return series
///
/// # Returns
///
/// * `Result<F>` - Portfolio beta
pub fn calculate_portfolio_beta<F: Float + Clone>(
    portfolio_returns: &Array1<F>,
    benchmark_returns: &Array1<F>,
) -> Result<F> {
    if portfolio_returns.len() != benchmark_returns.len() {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: portfolio_returns.len(),
            actual: benchmark_returns.len(),
        });
    }

    if portfolio_returns.len() < 2 {
        return Err(TimeSeriesError::InsufficientData {
            message: "Need at least 2 observations for beta calculation".to_string(),
            required: 2,
            actual: portfolio_returns.len(),
        });
    }

    let portfolio_mean = portfolio_returns.sum() / F::from(portfolio_returns.len()).unwrap();
    let benchmark_mean = benchmark_returns.sum() / F::from(benchmark_returns.len()).unwrap();

    let mut covariance = F::zero();
    let mut benchmark_variance = F::zero();

    for i in 0..portfolio_returns.len() {
        let portfolio_dev = portfolio_returns[i] - portfolio_mean;
        let benchmark_dev = benchmark_returns[i] - benchmark_mean;

        covariance = covariance + portfolio_dev * benchmark_dev;
        benchmark_variance = benchmark_variance + benchmark_dev.powi(2);
    }

    let n = F::from(portfolio_returns.len() - 1).unwrap();
    covariance = covariance / n;
    benchmark_variance = benchmark_variance / n;

    if benchmark_variance == F::zero() {
        Err(TimeSeriesError::InvalidInput(
            "Benchmark returns have zero variance".to_string(),
        ))
    } else {
        Ok(covariance / benchmark_variance)
    }
}

/// Calculate up/down capture ratios
///
/// Measures how well a portfolio captures upside vs downside market movements.
/// Values above 1.0 indicate the portfolio moves more than the benchmark.
///
/// # Arguments
///
/// * `portfolio_returns` - Portfolio return series
/// * `benchmark_returns` - Benchmark return series
///
/// # Returns
///
/// * `Result<(F, F)>` - (upside_capture, downside_capture) ratios
pub fn calculate_capture_ratios<F: Float + Clone>(
    portfolio_returns: &Array1<F>,
    benchmark_returns: &Array1<F>,
) -> Result<(F, F)> {
    if portfolio_returns.len() != benchmark_returns.len() {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: portfolio_returns.len(),
            actual: benchmark_returns.len(),
        });
    }

    let mut upside_portfolio = F::zero();
    let mut upside_benchmark = F::zero();
    let mut downside_portfolio = F::zero();
    let mut downside_benchmark = F::zero();
    let mut upside_count = 0;
    let mut downside_count = 0;

    for i in 0..portfolio_returns.len() {
        if benchmark_returns[i] > F::zero() {
            upside_portfolio = upside_portfolio + portfolio_returns[i];
            upside_benchmark = upside_benchmark + benchmark_returns[i];
            upside_count += 1;
        } else if benchmark_returns[i] < F::zero() {
            downside_portfolio = downside_portfolio + portfolio_returns[i];
            downside_benchmark = downside_benchmark + benchmark_returns[i];
            downside_count += 1;
        }
    }

    let upside_capture = if upside_count > 0 && upside_benchmark != F::zero() {
        (upside_portfolio / F::from(upside_count).unwrap())
            / (upside_benchmark / F::from(upside_count).unwrap())
    } else {
        F::zero()
    };

    let downside_capture = if downside_count > 0 && downside_benchmark != F::zero() {
        (downside_portfolio / F::from(downside_count).unwrap())
            / (downside_benchmark / F::from(downside_count).unwrap())
    } else {
        F::zero()
    };

    Ok((upside_capture, downside_capture))
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr1;

    #[test]
    fn test_calculate_portfolio_metrics() {
        let returns = arr1(&[0.01, -0.02, 0.015, -0.008, 0.012]);
        let prices = arr1(&[1000.0, 1010.0, 989.8, 1004.64, 996.6, 1008.6]);
        let risk_free_rate = 0.02;
        let periods_per_year = 252;

        let result =
            calculate_portfolio_metrics(&returns, &prices, risk_free_rate, periods_per_year);
        assert!(result.is_ok());

        let metrics = result.unwrap();
        assert!(metrics.is_valid());
        assert!(metrics.volatility > 0.0);
    }

    #[test]
    fn test_return_statistics() {
        let returns = arr1(&[0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.003, 0.008]);

        let result = calculate_return_statistics(&returns);
        assert!(result.is_ok());

        let (mean, std_dev, skewness, kurtosis) = result.unwrap();
        assert!(std_dev > 0.0);
        assert!(mean.is_finite());
        assert!(skewness.is_finite());
        assert!(kurtosis.is_finite());
    }

    #[test]
    fn test_tracking_error() {
        let portfolio = arr1(&[0.01, -0.02, 0.015, -0.008, 0.012]);
        let benchmark = arr1(&[0.008, -0.018, 0.012, -0.006, 0.010]);

        let result = calculate_tracking_error(&portfolio, &benchmark);
        assert!(result.is_ok());

        let te = result.unwrap();
        assert!(te >= 0.0);
        assert!(te.is_finite());
    }

    #[test]
    fn test_portfolio_beta() {
        let portfolio = arr1(&[0.01, -0.02, 0.015, -0.008, 0.012]);
        let benchmark = arr1(&[0.008, -0.018, 0.012, -0.006, 0.010]);

        let result = calculate_portfolio_beta(&portfolio, &benchmark);
        assert!(result.is_ok());

        let beta = result.unwrap();
        assert!(beta.is_finite());
    }

    #[test]
    fn test_capture_ratios() {
        let portfolio = arr1(&[0.01, -0.02, 0.015, -0.008, 0.012]);
        let benchmark = arr1(&[0.008, -0.018, 0.012, -0.006, 0.010]);

        let result = calculate_capture_ratios(&portfolio, &benchmark);
        assert!(result.is_ok());

        let (upside, downside) = result.unwrap();
        assert!(upside >= 0.0);
        assert!(downside >= 0.0);
    }

    #[test]
    fn test_dimension_mismatch() {
        let portfolio = arr1(&[0.01, -0.02]);
        let benchmark = arr1(&[0.008, -0.018, 0.012]);

        let result = calculate_tracking_error(&portfolio, &benchmark);
        assert!(result.is_err());
    }

    #[test]
    fn test_insufficient_data() {
        let returns = arr1(&[0.01]);

        let result = calculate_return_statistics(&returns);
        assert!(result.is_err());
    }
}
