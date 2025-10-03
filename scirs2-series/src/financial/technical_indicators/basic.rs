//! Basic technical indicators
//!
//! This module provides fundamental technical indicators commonly used in
//! financial analysis. These indicators help identify trends, momentum,
//! volatility, and volume patterns in price data.
//!
//! # Overview
//!
//! Technical indicators are mathematical calculations based on price, volume,
//! or open interest data. They help traders and analysts make informed decisions
//! by identifying patterns and trends in market data.
//!
//! # Categories
//!
//! - **Trend Indicators**: SMA, EMA, MACD - identify market direction
//! - **Momentum Indicators**: RSI, Stochastic, Williams %R - measure speed of price changes  
//! - **Volatility Indicators**: Bollinger Bands, ATR - measure price volatility
//! - **Volume Indicators**: OBV, MFI - analyze volume patterns
//!
//! # Examples
//!
//! ## Moving Averages
//! ```rust
//! use scirs2_series::financial::technical_indicators::basic::{sma, ema};
//! use scirs2_core::ndarray::array;
//!
//! let prices = array![10.0, 12.0, 13.0, 11.0, 14.0, 15.0];
//!
//! // Simple Moving Average with window of 3
//! let sma_values = sma(&prices, 3).unwrap();
//!
//! // Exponential Moving Average with alpha of 0.3
//! let ema_values = ema(&prices, 0.3).unwrap();
//! ```
//!
//! ## Momentum Oscillators
//! ```rust
//! use scirs2_series::financial::technical_indicators::basic::rsi;
//! use scirs2_core::ndarray::array;
//!
//! let prices = array![44.0, 44.25, 44.5, 43.75, 44.5, 45.0, 45.25, 45.5];
//! let rsi_values = rsi(&prices, 6).unwrap();
//! ```
//!
//! ## Bollinger Bands
//! ```rust
//! use scirs2_series::financial::technical_indicators::basic::bollinger_bands;
//! use scirs2_core::ndarray::array;
//!
//! let prices = array![20.0, 21.0, 19.5, 22.0, 21.5, 20.0, 19.0];
//! let (upper, middle, lower) = bollinger_bands(&prices, 5, 2.0).unwrap();
//! ```

use scirs2_core::ndarray::{s, Array1};
use scirs2_core::numeric::Float;

use crate::error::{Result, TimeSeriesError};

/// Simple Moving Average (SMA)
///
/// Calculates the arithmetic mean of prices over a specified number of periods.
/// The SMA is a lagging indicator that smooths out price action by creating
/// a constantly updated average price.
///
/// # Arguments
///
/// * `data` - Price data array
/// * `window` - Number of periods to average
///
/// # Returns
///
/// * `Result<Array1<F>>` - Array of SMA values
///
/// # Errors
///
/// * Returns error if window is zero or data has insufficient length
///
/// # Examples
///
/// ```rust
/// use scirs2_series::financial::technical_indicators::basic::sma;
/// use scirs2_core::ndarray::array;
///
/// let prices = array![10.0, 11.0, 12.0, 13.0, 14.0];
/// let sma_3 = sma(&prices, 3).unwrap();
/// // Returns: [11.0, 12.0, 13.0] (3-period averages)
/// ```
pub fn sma<F: Float + Clone>(data: &Array1<F>, window: usize) -> Result<Array1<F>> {
    if window == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "Window size must be positive".to_string(),
        ));
    }

    if data.len() < window {
        return Err(TimeSeriesError::InsufficientData {
            message: "Not enough data for SMA calculation".to_string(),
            required: window,
            actual: data.len(),
        });
    }

    let mut result = Array1::zeros(data.len() - window + 1);

    for i in 0..result.len() {
        let sum = data.slice(s![i..i + window]).sum();
        let window_f = F::from(window).unwrap();
        result[i] = sum / window_f;
    }

    Ok(result)
}

/// Exponential Moving Average (EMA)
///
/// Calculates the exponentially weighted moving average, giving more weight
/// to recent prices. The EMA reacts more quickly to recent price changes
/// than the SMA.
///
/// # Arguments
///
/// * `data` - Price data array
/// * `alpha` - Smoothing factor between 0 and 1 (higher values give more weight to recent prices)
///
/// # Returns
///
/// * `Result<Array1<F>>` - Array of EMA values
///
/// # Errors
///
/// * Returns error if data is empty or alpha is outside valid range
///
/// # Examples
///
/// ```rust
/// use scirs2_series::financial::technical_indicators::basic::ema;
/// use scirs2_core::ndarray::array;
///
/// let prices = array![10.0, 11.0, 12.0, 13.0, 14.0];
/// let ema_values = ema(&prices, 0.3).unwrap();
/// ```
pub fn ema<F: Float + Clone>(data: &Array1<F>, alpha: F) -> Result<Array1<F>> {
    if data.is_empty() {
        return Err(TimeSeriesError::InvalidInput(
            "Data cannot be empty".to_string(),
        ));
    }

    let zero = F::zero();
    let one = F::one();

    if alpha <= zero || alpha > one {
        return Err(TimeSeriesError::InvalidParameter {
            name: "alpha".to_string(),
            message: "Alpha must be between 0 and 1".to_string(),
        });
    }

    let mut result = Array1::zeros(data.len());
    result[0] = data[0];

    let one_minus_alpha = one - alpha;

    for i in 1..data.len() {
        result[i] = alpha * data[i] + one_minus_alpha * result[i - 1];
    }

    Ok(result)
}

/// Bollinger Bands
///
/// Calculates Bollinger Bands consisting of a middle line (SMA) and two bands
/// (upper and lower) positioned at a specified number of standard deviations
/// from the middle line. Used to identify overbought/oversold conditions.
///
/// # Arguments
///
/// * `data` - Price data array
/// * `window` - Period for moving average and standard deviation calculation
/// * `num_std` - Number of standard deviations for the bands
///
/// # Returns
///
/// * `Result<(Array1<F>, Array1<F>, Array1<F>)>` - Tuple of (upper_band, middle_line, lower_band)
///
/// # Examples
///
/// ```rust
/// use scirs2_series::financial::technical_indicators::basic::bollinger_bands;
/// use scirs2_core::ndarray::array;
///
/// let prices = array![20.0, 21.0, 19.5, 22.0, 21.5, 20.0, 19.0];
/// let (upper, middle, lower) = bollinger_bands(&prices, 5, 2.0).unwrap();
/// ```
pub fn bollinger_bands<F: Float + Clone>(
    data: &Array1<F>,
    window: usize,
    num_std: F,
) -> Result<(Array1<F>, Array1<F>, Array1<F>)> {
    let sma_values = sma(data, window)?;
    let mut upper = Array1::zeros(sma_values.len());
    let mut lower = Array1::zeros(sma_values.len());

    for i in 0..sma_values.len() {
        let slice = data.slice(s![i..i + window]);
        let mean = sma_values[i];

        // Calculate standard deviation
        let variance = slice
            .mapv(|x| {
                let diff = x - mean;
                diff * diff
            })
            .sum()
            / F::from(window).unwrap();

        let std_dev = variance.sqrt();

        upper[i] = mean + num_std * std_dev;
        lower[i] = mean - num_std * std_dev;
    }

    Ok((upper, sma_values, lower))
}

/// Relative Strength Index (RSI)
///
/// Measures the magnitude of recent price changes to evaluate overbought
/// or oversold conditions. RSI oscillates between 0 and 100, with values
/// above 70 typically indicating overbought conditions and below 30 indicating
/// oversold conditions.
///
/// # Arguments
///
/// * `data` - Price data array
/// * `period` - Number of periods for RSI calculation
///
/// # Returns
///
/// * `Result<Array1<F>>` - Array of RSI values
///
/// # Examples
///
/// ```rust
/// use scirs2_series::financial::technical_indicators::basic::rsi;
/// use scirs2_core::ndarray::array;
///
/// let prices = array![44.0, 44.25, 44.5, 43.75, 44.5, 45.0, 45.25, 45.5];
/// let rsi_values = rsi(&prices, 6).unwrap();
/// ```
pub fn rsi<F: Float + Clone>(data: &Array1<F>, period: usize) -> Result<Array1<F>> {
    if period == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "Period must be positive".to_string(),
        ));
    }

    if data.len() < period + 1 {
        return Err(TimeSeriesError::InsufficientData {
            message: "Not enough data for RSI calculation".to_string(),
            required: period + 1,
            actual: data.len(),
        });
    }

    // Calculate price changes
    let mut changes = Array1::zeros(data.len() - 1);
    for i in 0..changes.len() {
        changes[i] = data[i + 1] - data[i];
    }

    // Separate gains and losses
    let gains = changes.mapv(|x| if x > F::zero() { x } else { F::zero() });
    let losses = changes.mapv(|x| if x < F::zero() { -x } else { F::zero() });

    // Calculate average gains and losses
    let avg_gain = sma(&gains, period)?;
    let avg_loss = sma(&losses, period)?;

    // Calculate RSI
    let mut rsi = Array1::zeros(avg_gain.len());
    let hundred = F::from(100).unwrap();

    for i in 0..rsi.len() {
        if avg_loss[i] == F::zero() {
            rsi[i] = hundred;
        } else {
            let rs = avg_gain[i] / avg_loss[i];
            rsi[i] = hundred - (hundred / (F::one() + rs));
        }
    }

    Ok(rsi)
}

/// MACD (Moving Average Convergence Divergence)
///
/// A trend-following momentum indicator that shows the relationship between
/// two moving averages of a security's price. Consists of MACD line, signal
/// line, and histogram.
///
/// # Arguments
///
/// * `data` - Price data array
/// * `fast_period` - Period for fast EMA
/// * `slow_period` - Period for slow EMA
/// * `signal_period` - Period for signal line EMA
///
/// # Returns
///
/// * `Result<(Array1<F>, Array1<F>, Array1<F>)>` - Tuple of (macd_line, signal_line, histogram)
///
/// # Examples
///
/// ```rust
/// use scirs2_series::financial::technical_indicators::basic::macd;
/// use scirs2_core::ndarray::array;
///
/// let prices = array![12.0, 13.0, 14.0, 13.5, 15.0, 16.0, 15.5, 17.0];
/// let (macd_line, signal_line, histogram) = macd(&prices, 3, 6, 2).unwrap();
/// ```
pub fn macd<F: Float + Clone>(
    data: &Array1<F>,
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
) -> Result<(Array1<F>, Array1<F>, Array1<F>)> {
    if fast_period >= slow_period {
        return Err(TimeSeriesError::InvalidInput(
            "Fast period must be less than slow period".to_string(),
        ));
    }

    let fast_alpha = F::from(2.0).unwrap() / F::from(fast_period + 1).unwrap();
    let slow_alpha = F::from(2.0).unwrap() / F::from(slow_period + 1).unwrap();
    let signal_alpha = F::from(2.0).unwrap() / F::from(signal_period + 1).unwrap();

    let fast_ema = ema(data, fast_alpha)?;
    let slow_ema = ema(data, slow_alpha)?;

    // Calculate MACD line
    let macd_line = &fast_ema - &slow_ema;

    // Calculate signal line
    let signal_line = ema(&macd_line, signal_alpha)?;

    // Calculate histogram
    let histogram = &macd_line - &signal_line;

    Ok((macd_line, signal_line, histogram))
}

/// Stochastic Oscillator
///
/// Compares a closing price to its price range over a specific time period.
/// The oscillator consists of %K (fast stochastic) and %D (slow stochastic)
/// lines. Values range from 0 to 100.
///
/// # Arguments
///
/// * `high` - High price data
/// * `low` - Low price data  
/// * `close` - Closing price data
/// * `k_period` - Period for %K calculation
/// * `d_period` - Period for %D smoothing
///
/// # Returns
///
/// * `Result<(Array1<F>, Array1<F>)>` - Tuple of (%K, %D)
///
/// # Examples
///
/// ```rust
/// use scirs2_series::financial::technical_indicators::basic::stochastic;
/// use scirs2_core::ndarray::array;
///
/// let high = array![15.0, 16.0, 14.5, 17.0, 16.5];
/// let low = array![13.0, 14.0, 13.5, 15.0, 15.5];
/// let close = array![14.5, 15.5, 14.0, 16.0, 16.0];
/// let (k_percent, d_percent) = stochastic(&high, &low, &close, 3, 2).unwrap();
/// ```
pub fn stochastic<F: Float + Clone>(
    high: &Array1<F>,
    low: &Array1<F>,
    close: &Array1<F>,
    k_period: usize,
    d_period: usize,
) -> Result<(Array1<F>, Array1<F>)> {
    if high.len() != low.len() || low.len() != close.len() {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: high.len(),
            actual: close.len(),
        });
    }

    if high.len() < k_period {
        return Err(TimeSeriesError::InsufficientData {
            message: "Not enough data for stochastic calculation".to_string(),
            required: k_period,
            actual: high.len(),
        });
    }

    let mut k_percent = Array1::zeros(high.len() - k_period + 1);
    let hundred = F::from(100).unwrap();

    for i in 0..k_percent.len() {
        let period_high = high
            .slice(s![i..i + k_period])
            .iter()
            .cloned()
            .fold(F::neg_infinity(), F::max);
        let period_low = low
            .slice(s![i..i + k_period])
            .iter()
            .cloned()
            .fold(F::infinity(), F::min);

        let current_close = close[i + k_period - 1];

        if period_high == period_low {
            k_percent[i] = hundred;
        } else {
            k_percent[i] = hundred * (current_close - period_low) / (period_high - period_low);
        }
    }

    let d_percent = sma(&k_percent, d_period)?;

    Ok((k_percent, d_percent))
}

/// Average True Range (ATR)
///
/// Measures market volatility by calculating the average of true ranges over
/// a specified period. True range is the maximum of: current high-low,
/// abs(current high - previous close), abs(current low - previous close).
///
/// # Arguments
///
/// * `high` - High price data
/// * `low` - Low price data
/// * `close` - Closing price data
/// * `period` - Number of periods for averaging
///
/// # Returns
///
/// * `Result<Array1<F>>` - Array of ATR values
///
/// # Examples
///
/// ```rust
/// use scirs2_series::financial::technical_indicators::basic::atr;
/// use scirs2_core::ndarray::array;
///
/// let high = array![15.0, 16.0, 14.5, 17.0, 16.5];
/// let low = array![13.0, 14.0, 13.5, 15.0, 15.5];
/// let close = array![14.5, 15.5, 14.0, 16.0, 16.0];
/// let atr_values = atr(&high, &low, &close, 3).unwrap();
/// ```
pub fn atr<F: Float + Clone>(
    high: &Array1<F>,
    low: &Array1<F>,
    close: &Array1<F>,
    period: usize,
) -> Result<Array1<F>> {
    if high.len() != low.len() || low.len() != close.len() {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: high.len(),
            actual: close.len(),
        });
    }

    if high.len() < period + 1 {
        return Err(TimeSeriesError::InsufficientData {
            message: "Not enough data for ATR calculation".to_string(),
            required: period + 1,
            actual: high.len(),
        });
    }

    let mut true_ranges = Array1::zeros(high.len() - 1);

    for i in 1..high.len() {
        let tr1 = high[i] - low[i];
        let tr2 = (high[i] - close[i - 1]).abs();
        let tr3 = (low[i] - close[i - 1]).abs();

        true_ranges[i - 1] = tr1.max(tr2).max(tr3);
    }

    sma(&true_ranges, period)
}

/// Williams %R Oscillator
///
/// A momentum indicator that measures overbought and oversold levels.
/// Similar to the Stochastic Oscillator but uses a different scale (-100 to 0).
/// Values above -20 indicate overbought conditions, below -80 indicate oversold.
///
/// # Arguments
///
/// * `high` - High price data
/// * `low` - Low price data
/// * `close` - Closing price data  
/// * `period` - Number of periods for calculation
///
/// # Returns
///
/// * `Result<Array1<F>>` - Array of Williams %R values
///
/// # Examples
///
/// ```rust
/// use scirs2_series::financial::technical_indicators::basic::williams_r;
/// use scirs2_core::ndarray::array;
///
/// let high = array![15.0, 16.0, 14.5, 17.0, 16.5];
/// let low = array![13.0, 14.0, 13.5, 15.0, 15.5];
/// let close = array![14.5, 15.5, 14.0, 16.0, 16.0];
/// let williams_r_values = williams_r(&high, &low, &close, 3).unwrap();
/// ```
pub fn williams_r<F: Float + Clone>(
    high: &Array1<F>,
    low: &Array1<F>,
    close: &Array1<F>,
    period: usize,
) -> Result<Array1<F>> {
    if high.len() != low.len() || low.len() != close.len() {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: high.len(),
            actual: close.len(),
        });
    }

    if high.len() < period {
        return Err(TimeSeriesError::InsufficientData {
            message: "Not enough data for Williams %R calculation".to_string(),
            required: period,
            actual: high.len(),
        });
    }

    let mut williams_r = Array1::zeros(high.len() - period + 1);
    let hundred = F::from(100).unwrap();

    for i in 0..williams_r.len() {
        let period_high = high
            .slice(s![i..i + period])
            .iter()
            .cloned()
            .fold(F::neg_infinity(), F::max);
        let period_low = low
            .slice(s![i..i + period])
            .iter()
            .cloned()
            .fold(F::infinity(), F::min);

        let current_close = close[i + period - 1];

        if period_high == period_low {
            williams_r[i] = F::zero();
        } else {
            williams_r[i] =
                ((period_high - current_close) / (period_high - period_low)) * (-hundred);
        }
    }

    Ok(williams_r)
}

/// Commodity Channel Index (CCI)
///
/// Measures the variation of a security's price from its statistical mean.
/// CCI oscillates above and below zero. Positive values indicate prices are
/// above the average, negative values indicate they are below.
///
/// # Arguments
///
/// * `high` - High price data
/// * `low` - Low price data
/// * `close` - Closing price data
/// * `period` - Number of periods for calculation
///
/// # Returns
///
/// * `Result<Array1<F>>` - Array of CCI values
///
/// # Examples
///
/// ```rust
/// use scirs2_series::financial::technical_indicators::basic::cci;
/// use scirs2_core::ndarray::array;
///
/// let high = array![15.0, 16.0, 14.5, 17.0, 16.5];
/// let low = array![13.0, 14.0, 13.5, 15.0, 15.5];
/// let close = array![14.5, 15.5, 14.0, 16.0, 16.0];
/// let cci_values = cci(&high, &low, &close, 3).unwrap();
/// ```
pub fn cci<F: Float + Clone>(
    high: &Array1<F>,
    low: &Array1<F>,
    close: &Array1<F>,
    period: usize,
) -> Result<Array1<F>> {
    if high.len() != low.len() || low.len() != close.len() {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: high.len(),
            actual: close.len(),
        });
    }

    if high.len() < period {
        return Err(TimeSeriesError::InsufficientData {
            message: "Not enough data for CCI calculation".to_string(),
            required: period,
            actual: high.len(),
        });
    }

    // Calculate Typical Price
    let mut typical_price = Array1::zeros(high.len());
    let three = F::from(3).unwrap();

    for i in 0..high.len() {
        typical_price[i] = (high[i] + low[i] + close[i]) / three;
    }

    // Calculate SMA of typical price
    let sma_tp = sma(&typical_price, period)?;

    // Calculate mean deviation
    let mut cci = Array1::zeros(sma_tp.len());
    let constant = F::from(0.015).unwrap();

    for i in 0..cci.len() {
        let slice = typical_price.slice(s![i..i + period]);
        let mean = sma_tp[i];

        let mean_deviation = slice.mapv(|x| (x - mean).abs()).sum() / F::from(period).unwrap();

        if mean_deviation != F::zero() {
            cci[i] = (typical_price[i + period - 1] - mean) / (constant * mean_deviation);
        }
    }

    Ok(cci)
}

/// On-Balance Volume (OBV)
///
/// Combines price and volume to show how money may be flowing into or out
/// of a security. If closing price is higher than previous close, volume
/// is added; if lower, volume is subtracted.
///
/// # Arguments
///
/// * `close` - Closing price data
/// * `volume` - Volume data
///
/// # Returns
///
/// * `Result<Array1<F>>` - Array of OBV values
///
/// # Examples
///
/// ```rust
/// use scirs2_series::financial::technical_indicators::basic::obv;
/// use scirs2_core::ndarray::array;
///
/// let close = array![10.0, 10.5, 10.2, 10.8, 11.0];
/// let volume = array![1000.0, 1200.0, 800.0, 1500.0, 2000.0];
/// let obv_values = obv(&close, &volume).unwrap();
/// ```
pub fn obv<F: Float + Clone>(close: &Array1<F>, volume: &Array1<F>) -> Result<Array1<F>> {
    if close.len() != volume.len() {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: close.len(),
            actual: volume.len(),
        });
    }

    if close.len() < 2 {
        return Err(TimeSeriesError::InsufficientData {
            message: "Need at least 2 data points for OBV".to_string(),
            required: 2,
            actual: close.len(),
        });
    }

    let mut obv = Array1::zeros(close.len());
    obv[0] = volume[0];

    for i in 1..close.len() {
        if close[i] > close[i - 1] {
            obv[i] = obv[i - 1] + volume[i];
        } else if close[i] < close[i - 1] {
            obv[i] = obv[i - 1] - volume[i];
        } else {
            obv[i] = obv[i - 1];
        }
    }

    Ok(obv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr1;

    #[test]
    fn test_sma() {
        let data = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = sma(&data, 3).unwrap();
        let expected = arr1(&[2.0, 3.0, 4.0]);

        for (actual, expected) in result.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_ema() {
        let data = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = ema(&data, 0.5).unwrap();

        assert_eq!(result[0], 1.0);
        assert!((result[1] - 1.5).abs() < 1e-10); // 0.5*2 + 0.5*1
        assert!(result.len() == data.len());
    }

    #[test]
    fn test_rsi() {
        let data = arr1(&[44.0, 44.25, 44.5, 43.75, 44.5, 45.0, 45.25, 45.5]);
        let result = rsi(&data, 3);
        assert!(result.is_ok());

        let rsi_values = result.unwrap();
        // All values should be between 0 and 100
        for &value in rsi_values.iter() {
            assert!(value >= 0.0 && value <= 100.0);
        }
    }

    #[test]
    fn test_bollinger_bands() {
        let data = arr1(&[20.0, 21.0, 19.5, 22.0, 21.5]);
        let (upper, middle, lower) = bollinger_bands(&data, 3, 2.0).unwrap();

        // Upper band should be above middle, middle above lower
        for i in 0..upper.len() {
            assert!(upper[i] > middle[i]);
            assert!(middle[i] > lower[i]);
        }
    }

    #[test]
    fn test_macd() {
        let data = arr1(&[12.0, 13.0, 14.0, 13.5, 15.0, 16.0, 15.5, 17.0]);
        let result = macd(&data, 3, 6, 2);
        assert!(result.is_ok());

        let (macd_line, signal_line, histogram) = result.unwrap();
        assert_eq!(macd_line.len(), data.len());
        assert_eq!(signal_line.len(), data.len());
        assert_eq!(histogram.len(), data.len());
    }

    #[test]
    fn test_atr() {
        let high = arr1(&[15.0, 16.0, 14.5, 17.0, 16.5]);
        let low = arr1(&[13.0, 14.0, 13.5, 15.0, 15.5]);
        let close = arr1(&[14.5, 15.5, 14.0, 16.0, 16.0]);

        let result = atr(&high, &low, &close, 3);
        assert!(result.is_ok());

        let atr_values = result.unwrap();
        // All ATR values should be positive
        for &value in atr_values.iter() {
            assert!(value >= 0.0);
        }
    }

    #[test]
    fn test_insufficient_data() {
        let data = arr1(&[1.0, 2.0]);
        let result = sma(&data, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_parameters() {
        let data = arr1(&[1.0, 2.0, 3.0]);

        // Zero window
        let result = sma(&data, 0);
        assert!(result.is_err());

        // Invalid alpha for EMA
        let result = ema(&data, 1.5);
        assert!(result.is_err());
    }
}
