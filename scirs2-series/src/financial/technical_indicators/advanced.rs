//! Advanced technical indicators
//!
//! This module provides sophisticated technical indicators that go beyond basic
//! moving averages and oscillators. These indicators offer more complex analysis
//! capabilities including multi-dimensional analysis, adaptive calculations,
//! and comprehensive market structure evaluation.
//!
//! # Overview
//!
//! Advanced technical indicators typically provide:
//! - **Multiple output signals**: Complex indicators that generate several complementary signals
//! - **Adaptive behavior**: Indicators that adjust their sensitivity based on market conditions
//! - **Multi-timeframe analysis**: Indicators designed for cross-timeframe analysis
//! - **Volume-price integration**: Indicators that combine price and volume data
//! - **Market structure analysis**: Indicators that reveal market microstructure patterns
//!
//! # Categories
//!
//! - **Trend Analysis**: ADX, Parabolic SAR, Ichimoku Cloud, Aroon
//! - **Volatility Indicators**: Enhanced Bollinger Bands with bandwidth analysis
//! - **Volume Indicators**: MFI, VWAP, Chaikin Oscillator
//! - **Momentum Oscillators**: Enhanced Stochastic with configurable smoothing
//! - **Adaptive Indicators**: KAMA (Kaufman's Adaptive Moving Average)
//! - **Support/Resistance**: Fibonacci Retracement Levels
//!
//! # Examples
//!
//! ## Enhanced Bollinger Bands
//! ```rust
//! use scirs2_series::financial::technical_indicators::advanced::{BollingerBandsConfig, bollinger_bands, MovingAverageType};
//! use ndarray::array;
//!
//! let prices = array![20.0, 21.0, 19.5, 22.0, 21.5, 20.0, 19.0, 23.0, 22.5, 21.0];
//! let config = BollingerBandsConfig {
//!     period: 5,
//!     std_dev_multiplier: 2.0,
//!     ma_type: MovingAverageType::Simple,
//! };
//!
//! let bands = bollinger_bands(&prices, &config).unwrap();
//! println!("Bandwidth: {:?}", bands.bandwidth);
//! println!("%B Position: {:?}", bands.percent_b);
//! ```
//!
//! ## Ichimoku Cloud Analysis
//! ```rust
//! use scirs2_series::financial::technical_indicators::advanced::{IchimokuConfig, ichimoku_cloud};
//! use ndarray::Array1;
//!
//! let high = (0..60).map(|i| 15.0 + (i as f64 * 0.1).sin() * 2.0).collect::<Vec<f64>>();
//! let low = (0..60).map(|i| 13.0 + (i as f64 * 0.1).sin() * 2.0).collect::<Vec<f64>>();
//! let close = (0..60).map(|i| 14.0 + (i as f64 * 0.1).sin() * 2.0).collect::<Vec<f64>>();
//! let high = Array1::from_vec(high);
//! let low = Array1::from_vec(low);
//! let close = Array1::from_vec(close);
//! let config = IchimokuConfig::default();
//!
//! let cloud = ichimoku_cloud(&high, &low, &close, &config).unwrap();
//! ```
//!
//! ## Adaptive Moving Average
//! ```rust
//! use scirs2_series::financial::technical_indicators::advanced::kama;
//! use ndarray::array;
//!
//! let prices = array![10.0, 12.0, 11.0, 13.0, 15.0, 14.0, 16.0, 18.0];
//! let kama_values = kama(&prices, 5, 2, 30).unwrap();
//! ```

use ndarray::{s, Array1};
use num_traits::Float;

use crate::error::{Result, TimeSeriesError};

/// Moving average types for advanced indicators
#[derive(Debug, Clone)]
pub enum MovingAverageType {
    /// Simple moving average
    Simple,
    /// Exponential moving average
    Exponential,
    /// Weighted moving average
    Weighted,
}

/// Bollinger Bands configuration
#[derive(Debug, Clone)]
pub struct BollingerBandsConfig {
    /// Period for moving average and standard deviation calculation
    pub period: usize,
    /// Number of standard deviations for the bands
    pub std_dev_multiplier: f64,
    /// Type of moving average to use
    pub ma_type: MovingAverageType,
}

impl Default for BollingerBandsConfig {
    fn default() -> Self {
        Self {
            period: 20,
            std_dev_multiplier: 2.0,
            ma_type: MovingAverageType::Simple,
        }
    }
}

/// Enhanced Bollinger Bands result with additional analysis
#[derive(Debug, Clone)]
pub struct BollingerBands<F: Float> {
    /// Upper band
    pub upper_band: Array1<F>,
    /// Middle band (moving average)
    pub middle_band: Array1<F>,
    /// Lower band
    pub lower_band: Array1<F>,
    /// Bandwidth (upper - lower) / middle
    pub bandwidth: Array1<F>,
    /// %B indicator (position within bands)
    pub percent_b: Array1<F>,
}

/// Calculate Enhanced Bollinger Bands
///
/// This advanced implementation provides additional metrics including bandwidth
/// and %B position indicator, with configurable moving average types.
///
/// # Arguments
///
/// * `prices` - Price data array
/// * `config` - Bollinger Bands configuration
///
/// # Returns
///
/// * `Result<BollingerBands<F>>` - Enhanced Bollinger Bands with analysis metrics
///
/// # Examples
///
/// ```rust
/// use scirs2_series::financial::technical_indicators::advanced::{BollingerBandsConfig, bollinger_bands};
/// use ndarray::array;
///
/// let prices = array![20.0, 21.0, 19.5, 22.0, 21.5, 20.0, 19.0, 23.0, 22.5, 21.0, 20.5, 22.5, 23.0, 21.5, 20.0, 22.0, 21.0, 19.5, 21.5, 22.0, 20.5];
/// let config = BollingerBandsConfig::default();
/// let bands = bollinger_bands(&prices, &config).unwrap();
/// ```
pub fn bollinger_bands<F: Float + Clone>(
    prices: &Array1<F>,
    config: &BollingerBandsConfig,
) -> Result<BollingerBands<F>> {
    if prices.len() < config.period {
        return Err(TimeSeriesError::InsufficientData {
            message: "Not enough data for Bollinger Bands".to_string(),
            required: config.period,
            actual: prices.len(),
        });
    }

    let output_len = prices.len() - config.period + 1;
    let mut upper_band = Array1::zeros(output_len);
    let mut middle_band = Array1::zeros(output_len);
    let mut lower_band = Array1::zeros(output_len);
    let mut bandwidth = Array1::zeros(output_len);
    let mut percent_b = Array1::zeros(output_len);

    let std_multiplier = F::from(config.std_dev_multiplier).unwrap();

    for i in 0..output_len {
        let window = prices.slice(s![i..i + config.period]);

        // Calculate moving average
        let ma = match config.ma_type {
            MovingAverageType::Simple => window.sum() / F::from(config.period).unwrap(),
            MovingAverageType::Exponential => {
                let alpha = F::from(2.0).unwrap() / F::from(config.period + 1).unwrap();
                let mut ema = window[0];
                for &price in window.iter().skip(1) {
                    ema = alpha * price + (F::one() - alpha) * ema;
                }
                ema
            }
            MovingAverageType::Weighted => {
                let mut sum = F::zero();
                let mut weight_sum = F::zero();
                for (j, &price) in window.iter().enumerate() {
                    let weight = F::from(j + 1).unwrap();
                    sum = sum + weight * price;
                    weight_sum = weight_sum + weight;
                }
                sum / weight_sum
            }
        };

        // Calculate standard deviation
        let variance = window
            .iter()
            .map(|&price| (price - ma).powi(2))
            .fold(F::zero(), |acc, x| acc + x)
            / F::from(config.period).unwrap();
        let std_dev = variance.sqrt();

        middle_band[i] = ma;
        upper_band[i] = ma + std_multiplier * std_dev;
        lower_band[i] = ma - std_multiplier * std_dev;

        // Calculate bandwidth
        bandwidth[i] = if ma > F::zero() {
            (upper_band[i] - lower_band[i]) / ma
        } else {
            F::zero()
        };

        // Calculate %B
        let current_price = prices[i + config.period - 1];
        percent_b[i] = if upper_band[i] != lower_band[i] {
            (current_price - lower_band[i]) / (upper_band[i] - lower_band[i])
        } else {
            F::from(0.5).unwrap()
        };
    }

    Ok(BollingerBands {
        upper_band,
        middle_band,
        lower_band,
        bandwidth,
        percent_b,
    })
}

/// Stochastic Oscillator configuration
#[derive(Debug, Clone)]
pub struct StochasticConfig {
    /// %K period
    pub k_period: usize,
    /// %D period (smoothing)
    pub d_period: usize,
    /// %D smoothing type
    pub d_smoothing: MovingAverageType,
}

impl Default for StochasticConfig {
    fn default() -> Self {
        Self {
            k_period: 14,
            d_period: 3,
            d_smoothing: MovingAverageType::Simple,
        }
    }
}

/// Enhanced Stochastic Oscillator result
#[derive(Debug, Clone)]
pub struct StochasticOscillator<F: Float> {
    /// %K line (fast stochastic)
    pub percent_k: Array1<F>,
    /// %D line (slow stochastic)
    pub percent_d: Array1<F>,
}

/// Calculate Enhanced Stochastic Oscillator
///
/// This implementation provides configurable smoothing for the %D line
/// and supports different moving average types.
///
/// # Arguments
///
/// * `high` - High price data
/// * `low` - Low price data
/// * `close` - Closing price data
/// * `config` - Stochastic configuration
///
/// # Returns
///
/// * `Result<StochasticOscillator<F>>` - Enhanced stochastic oscillator
pub fn stochastic_oscillator<F: Float + Clone>(
    high: &Array1<F>,
    low: &Array1<F>,
    close: &Array1<F>,
    config: &StochasticConfig,
) -> Result<StochasticOscillator<F>> {
    if high.len() != low.len() || low.len() != close.len() {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: high.len(),
            actual: close.len(),
        });
    }

    if high.len() < config.k_period {
        return Err(TimeSeriesError::InsufficientData {
            message: "Not enough data for Stochastic Oscillator".to_string(),
            required: config.k_period,
            actual: high.len(),
        });
    }

    let k_output_len = high.len() - config.k_period + 1;
    let mut percent_k = Array1::zeros(k_output_len);

    // Calculate %K
    for i in 0..k_output_len {
        let window_start = i;
        let window_end = i + config.k_period;

        let highest_high = high
            .slice(s![window_start..window_end])
            .iter()
            .fold(F::neg_infinity(), |acc, &x| acc.max(x));
        let lowest_low = low
            .slice(s![window_start..window_end])
            .iter()
            .fold(F::infinity(), |acc, &x| acc.min(x));

        let current_close = close[window_end - 1];

        percent_k[i] = if highest_high != lowest_low {
            F::from(100.0).unwrap() * (current_close - lowest_low) / (highest_high - lowest_low)
        } else {
            F::from(50.0).unwrap()
        };
    }

    // Calculate %D (smoothed %K)
    let d_output_len = if k_output_len >= config.d_period {
        k_output_len - config.d_period + 1
    } else {
        0
    };

    let mut percent_d = Array1::zeros(d_output_len);

    for i in 0..d_output_len {
        let k_window = percent_k.slice(s![i..i + config.d_period]);
        percent_d[i] = match config.d_smoothing {
            MovingAverageType::Simple => k_window.sum() / F::from(config.d_period).unwrap(),
            MovingAverageType::Exponential => {
                let alpha = F::from(2.0).unwrap() / F::from(config.d_period + 1).unwrap();
                let mut ema = k_window[0];
                for &k_val in k_window.iter().skip(1) {
                    ema = alpha * k_val + (F::one() - alpha) * ema;
                }
                ema
            }
            MovingAverageType::Weighted => {
                let mut sum = F::zero();
                let mut weight_sum = F::zero();
                for (j, &k_val) in k_window.iter().enumerate() {
                    let weight = F::from(j + 1).unwrap();
                    sum = sum + weight * k_val;
                    weight_sum = weight_sum + weight;
                }
                sum / weight_sum
            }
        };
    }

    Ok(StochasticOscillator {
        percent_k,
        percent_d,
    })
}

/// Ichimoku Cloud configuration
#[derive(Debug, Clone)]
pub struct IchimokuConfig {
    /// Tenkan-sen period (conversion line)
    pub tenkan_period: usize,
    /// Kijun-sen period (base line)
    pub kijun_period: usize,
    /// Senkou Span B period
    pub senkou_b_period: usize,
    /// Displacement for Senkou spans
    pub displacement: usize,
}

impl Default for IchimokuConfig {
    fn default() -> Self {
        Self {
            tenkan_period: 9,
            kijun_period: 26,
            senkou_b_period: 52,
            displacement: 26,
        }
    }
}

/// Ichimoku Cloud result
#[derive(Debug, Clone)]
pub struct IchimokuCloud<F: Float> {
    /// Tenkan-sen (Conversion Line)
    pub tenkan_sen: Array1<F>,
    /// Kijun-sen (Base Line)
    pub kijun_sen: Array1<F>,
    /// Chikou Span (Lagging Span)
    pub chikou_span: Array1<F>,
    /// Senkou Span A (Leading Span A)
    pub senkou_span_a: Array1<F>,
    /// Senkou Span B (Leading Span B)
    pub senkou_span_b: Array1<F>,
}

/// Calculate Ichimoku Cloud
///
/// A comprehensive trend-following indicator that provides multiple signals
/// for trend direction, support/resistance levels, and momentum.
///
/// # Arguments
///
/// * `high` - High price data
/// * `low` - Low price data
/// * `close` - Closing price data
/// * `config` - Ichimoku configuration
///
/// # Returns
///
/// * `Result<IchimokuCloud<F>>` - Complete Ichimoku Cloud analysis
pub fn ichimoku_cloud<F: Float + Clone>(
    high: &Array1<F>,
    low: &Array1<F>,
    close: &Array1<F>,
    config: &IchimokuConfig,
) -> Result<IchimokuCloud<F>> {
    if high.len() != low.len() || low.len() != close.len() {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: high.len(),
            actual: close.len(),
        });
    }

    let min_length = config.senkou_b_period.max(config.displacement);
    if high.len() < min_length {
        return Err(TimeSeriesError::InsufficientData {
            message: "Not enough data for Ichimoku Cloud".to_string(),
            required: min_length,
            actual: high.len(),
        });
    }

    let n = high.len();

    // Helper function to calculate highest high and lowest low midpoint
    let calculate_hl_midpoint = |period: usize, start_idx: usize| -> F {
        let end_idx = (start_idx + period).min(n);
        let high_slice = high.slice(s![start_idx..end_idx]);
        let low_slice = low.slice(s![start_idx..end_idx]);
        let highest = high_slice
            .iter()
            .fold(F::neg_infinity(), |acc, &x| acc.max(x));
        let lowest = low_slice.iter().fold(F::infinity(), |acc, &x| acc.min(x));
        (highest + lowest) / F::from(2.0).unwrap()
    };

    // Calculate Tenkan-sen
    let mut tenkan_sen = Array1::zeros(n);
    for i in 0..n {
        if i + 1 >= config.tenkan_period {
            let start = i + 1 - config.tenkan_period;
            tenkan_sen[i] = calculate_hl_midpoint(config.tenkan_period, start);
        } else {
            tenkan_sen[i] = calculate_hl_midpoint(i + 1, 0);
        }
    }

    // Calculate Kijun-sen
    let mut kijun_sen = Array1::zeros(n);
    for i in 0..n {
        if i + 1 >= config.kijun_period {
            let start = i + 1 - config.kijun_period;
            kijun_sen[i] = calculate_hl_midpoint(config.kijun_period, start);
        } else {
            kijun_sen[i] = calculate_hl_midpoint(i + 1, 0);
        }
    }

    // Calculate Chikou Span (displaced close)
    let mut chikou_span = Array1::zeros(n);
    for i in 0..n {
        chikou_span[i] = close[i];
    }

    // Calculate Senkou Span A
    let mut senkou_span_a = Array1::zeros(n);
    for i in 0..n {
        senkou_span_a[i] = (tenkan_sen[i] + kijun_sen[i]) / F::from(2.0).unwrap();
    }

    // Calculate Senkou Span B
    let mut senkou_span_b = Array1::zeros(n);
    for i in 0..n {
        if i + 1 >= config.senkou_b_period {
            let start = i + 1 - config.senkou_b_period;
            senkou_span_b[i] = calculate_hl_midpoint(config.senkou_b_period, start);
        } else {
            senkou_span_b[i] = calculate_hl_midpoint(i + 1, 0);
        }
    }

    Ok(IchimokuCloud {
        tenkan_sen,
        kijun_sen,
        chikou_span,
        senkou_span_a,
        senkou_span_b,
    })
}

/// Average Directional Index (ADX)
///
/// Measures trend strength regardless of direction. Values above 25 typically
/// indicate strong trends, while values below 20 suggest weak or sideways trends.
///
/// # Arguments
///
/// * `high` - High price data
/// * `low` - Low price data
/// * `close` - Closing price data
/// * `period` - Calculation period
///
/// # Returns
///
/// * `Result<Array1<F>>` - ADX values (0-100)
pub fn adx<F: Float + Clone + num_traits::FromPrimitive>(
    high: &Array1<F>,
    low: &Array1<F>,
    close: &Array1<F>,
    period: usize,
) -> Result<Array1<F>> {
    if high.len() != low.len() || high.len() != close.len() {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: high.len(),
            actual: std::cmp::min(low.len(), close.len()),
        });
    }

    if high.len() < period + 1 {
        return Err(TimeSeriesError::InsufficientData {
            message: "Not enough data for ADX calculation".to_string(),
            required: period + 1,
            actual: high.len(),
        });
    }

    let n = high.len();
    let mut dx = Array1::zeros(n - 1);
    let mut adx = Array1::zeros(n - period);

    // Calculate True Range and Directional Movement
    for i in 1..n {
        let tr1 = high[i] - low[i];
        let tr2 = (high[i] - close[i - 1]).abs();
        let tr3 = (low[i] - close[i - 1]).abs();
        let true_range = tr1.max(tr2).max(tr3);

        let dm_plus = if high[i] > high[i - 1] && (high[i] - high[i - 1]) > (low[i - 1] - low[i]) {
            high[i] - high[i - 1]
        } else {
            F::zero()
        };

        let dm_minus = if low[i] < low[i - 1] && (low[i - 1] - low[i]) > (high[i] - high[i - 1]) {
            low[i - 1] - low[i]
        } else {
            F::zero()
        };

        let di_plus = if true_range > F::zero() {
            dm_plus / true_range
        } else {
            F::zero()
        };

        let di_minus = if true_range > F::zero() {
            dm_minus / true_range
        } else {
            F::zero()
        };

        dx[i - 1] = if di_plus + di_minus > F::zero() {
            (di_plus - di_minus).abs() / (di_plus + di_minus)
        } else {
            F::zero()
        };
    }

    // Calculate ADX using exponential moving average
    let alpha = F::from(2.0).unwrap() / F::from(period + 1).unwrap();
    let mut ema_dx = dx[0];

    for i in 0..adx.len() {
        if i == 0 {
            ema_dx = dx.slice(s![0..period]).sum() / F::from(period).unwrap();
        } else {
            ema_dx = alpha * dx[i + period - 1] + (F::one() - alpha) * ema_dx;
        }
        adx[i] = ema_dx * F::from(100.0).unwrap();
    }

    Ok(adx)
}

/// Parabolic Stop and Reverse (SAR)
///
/// A trend-following indicator that provides potential reversal points.
/// The SAR is designed to give exit points for long or short positions.
///
/// # Arguments
///
/// * `high` - High price data
/// * `low` - Low price data
/// * `acceleration_factor` - Initial acceleration factor (typically 0.02)
/// * `max_acceleration` - Maximum acceleration factor (typically 0.2)
///
/// # Returns
///
/// * `Result<Array1<F>>` - Parabolic SAR values
pub fn parabolic_sar<F: Float + Clone>(
    high: &Array1<F>,
    low: &Array1<F>,
    acceleration_factor: F,
    max_acceleration: F,
) -> Result<Array1<F>> {
    if high.len() != low.len() {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: high.len(),
            actual: low.len(),
        });
    }

    if high.len() < 3 {
        return Err(TimeSeriesError::InsufficientData {
            message: "Need at least 3 data points for Parabolic SAR".to_string(),
            required: 3,
            actual: high.len(),
        });
    }

    let n = high.len();
    let mut sar = Array1::zeros(n);
    let mut ep = high[0]; // Extreme point
    let mut af = acceleration_factor; // Acceleration factor
    let mut is_uptrend = true;

    sar[0] = low[0];
    sar[1] = low[0];

    for i in 2..n {
        let prev_sar = sar[i - 1];

        if is_uptrend {
            // Calculate SAR for uptrend
            sar[i] = prev_sar + af * (ep - prev_sar);

            // Check for trend reversal
            if low[i] <= sar[i] || low[i - 1] <= sar[i] {
                // Trend reversal to downtrend
                is_uptrend = false;
                sar[i] = ep;
                ep = low[i];
                af = acceleration_factor;
            } else {
                // Continue uptrend
                if high[i] > ep {
                    ep = high[i];
                    af = (af + acceleration_factor).min(max_acceleration);
                }

                // SAR should not be above previous two lows
                if i >= 2 {
                    sar[i] = sar[i].min(low[i - 1]).min(low[i - 2]);
                } else if i >= 1 {
                    sar[i] = sar[i].min(low[i - 1]);
                }
            }
        } else {
            // Calculate SAR for downtrend
            sar[i] = prev_sar + af * (ep - prev_sar);

            // Check for trend reversal
            if high[i] >= sar[i] || high[i - 1] >= sar[i] {
                // Trend reversal to uptrend
                is_uptrend = true;
                sar[i] = ep;
                ep = high[i];
                af = acceleration_factor;
            } else {
                // Continue downtrend
                if low[i] < ep {
                    ep = low[i];
                    af = (af + acceleration_factor).min(max_acceleration);
                }

                // SAR should not be below previous two highs
                if i >= 2 {
                    sar[i] = sar[i].max(high[i - 1]).max(high[i - 2]);
                } else if i >= 1 {
                    sar[i] = sar[i].max(high[i - 1]);
                }
            }
        }
    }

    Ok(sar)
}

/// Money Flow Index (MFI)
///
/// A volume-weighted version of RSI that measures buying and selling pressure.
/// Values above 80 typically indicate overbought conditions, while values
/// below 20 indicate oversold conditions.
///
/// # Arguments
///
/// * `high` - High price data
/// * `low` - Low price data
/// * `close` - Closing price data
/// * `volume` - Volume data
/// * `period` - Calculation period
///
/// # Returns
///
/// * `Result<Array1<F>>` - MFI values (0-100)
pub fn mfi<F: Float + Clone + std::iter::Sum>(
    high: &Array1<F>,
    low: &Array1<F>,
    close: &Array1<F>,
    volume: &Array1<F>,
    period: usize,
) -> Result<Array1<F>> {
    if high.len() != low.len() || low.len() != close.len() || close.len() != volume.len() {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: high.len(),
            actual: volume.len(),
        });
    }

    if high.len() < period + 1 {
        return Err(TimeSeriesError::InsufficientData {
            message: "Not enough data for MFI calculation".to_string(),
            required: period + 1,
            actual: high.len(),
        });
    }

    let n = high.len();
    let mut typical_prices = Array1::zeros(n);
    let mut money_flows = Array1::zeros(n - 1);
    let three = F::from(3.0).unwrap();

    // Calculate typical prices
    for i in 0..n {
        typical_prices[i] = (high[i] + low[i] + close[i]) / three;
    }

    // Calculate money flows
    for i in 1..n {
        let raw_money_flow = typical_prices[i] * volume[i];
        money_flows[i - 1] = if typical_prices[i] > typical_prices[i - 1] {
            raw_money_flow // Positive money flow
        } else if typical_prices[i] < typical_prices[i - 1] {
            -raw_money_flow // Negative money flow
        } else {
            F::zero() // Neutral
        };
    }

    let mut mfi = Array1::zeros(money_flows.len() - period + 1);
    let hundred = F::from(100.0).unwrap();

    for i in 0..mfi.len() {
        let slice = money_flows.slice(s![i..i + period]);
        let positive_flow: F = slice.iter().filter(|&&x| x > F::zero()).cloned().sum();
        let negative_flow: F = slice.iter().filter(|&&x| x < F::zero()).map(|&x| -x).sum();

        if negative_flow > F::zero() {
            let money_ratio = positive_flow / negative_flow;
            mfi[i] = hundred - (hundred / (F::one() + money_ratio));
        } else {
            mfi[i] = hundred;
        }
    }

    Ok(mfi)
}

/// Aroon Oscillator
///
/// Measures the time since the highest high and lowest low, helping
/// identify trend changes and strength. Returns (Aroon Up, Aroon Down, Oscillator).
///
/// # Arguments
///
/// * `high` - High price data
/// * `low` - Low price data
/// * `period` - Calculation period
///
/// # Returns
///
/// * `Result<(Array1<F>, Array1<F>, Array1<F>)>` - (Aroon Up, Aroon Down, Oscillator)
pub fn aroon<F: Float + Clone>(
    high: &Array1<F>,
    low: &Array1<F>,
    period: usize,
) -> Result<(Array1<F>, Array1<F>, Array1<F>)> {
    if high.len() != low.len() {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: high.len(),
            actual: low.len(),
        });
    }

    if high.len() < period {
        return Err(TimeSeriesError::InsufficientData {
            message: "Not enough data for Aroon calculation".to_string(),
            required: period,
            actual: high.len(),
        });
    }

    let n = high.len();
    let result_len = n - period + 1;
    let mut aroon_up = Array1::zeros(result_len);
    let mut aroon_down = Array1::zeros(result_len);
    let hundred = F::from(100.0).unwrap();
    let period_f = F::from(period).unwrap();

    for i in 0..result_len {
        let slice_high = high.slice(s![i..i + period]);
        let slice_low = low.slice(s![i..i + period]);

        // Find highest high and lowest low positions
        let max_pos = slice_high
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        let min_pos = slice_low
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        // Calculate Aroon Up and Aroon Down
        aroon_up[i] = hundred * (period_f - F::from(period - 1 - max_pos).unwrap()) / period_f;
        aroon_down[i] = hundred * (period_f - F::from(period - 1 - min_pos).unwrap()) / period_f;
    }

    // Calculate Aroon Oscillator
    let aroon_oscillator = &aroon_up - &aroon_down;

    Ok((aroon_up, aroon_down, aroon_oscillator))
}

/// Volume Weighted Average Price (VWAP)
///
/// A trading benchmark that gives the average price weighted by volume.
/// Often used to assess the quality of trade execution.
///
/// # Arguments
///
/// * `high` - High price data
/// * `low` - Low price data
/// * `close` - Closing price data
/// * `volume` - Volume data
///
/// # Returns
///
/// * `Result<Array1<F>>` - VWAP values
pub fn vwap<F: Float + Clone>(
    high: &Array1<F>,
    low: &Array1<F>,
    close: &Array1<F>,
    volume: &Array1<F>,
) -> Result<Array1<F>> {
    if high.len() != low.len() || low.len() != close.len() || close.len() != volume.len() {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: high.len(),
            actual: volume.len(),
        });
    }

    let n = high.len();
    let mut vwap = Array1::zeros(n);
    let mut cumulative_pv = F::zero();
    let mut cumulative_volume = F::zero();
    let three = F::from(3.0).unwrap();

    for i in 0..n {
        let typical_price = (high[i] + low[i] + close[i]) / three;
        let pv = typical_price * volume[i];

        cumulative_pv = cumulative_pv + pv;
        cumulative_volume = cumulative_volume + volume[i];

        if cumulative_volume > F::zero() {
            vwap[i] = cumulative_pv / cumulative_volume;
        } else {
            vwap[i] = typical_price;
        }
    }

    Ok(vwap)
}

/// Chaikin Oscillator
///
/// Combines price and volume to measure the accumulation/distribution
/// of a security. Uses the difference between fast and slow EMAs of
/// the Accumulation/Distribution line.
///
/// # Arguments
///
/// * `high` - High price data
/// * `low` - Low price data
/// * `close` - Closing price data
/// * `volume` - Volume data
/// * `fast_period` - Fast EMA period
/// * `slow_period` - Slow EMA period
///
/// # Returns
///
/// * `Result<Array1<F>>` - Chaikin Oscillator values
pub fn chaikin_oscillator<F: Float + Clone>(
    high: &Array1<F>,
    low: &Array1<F>,
    close: &Array1<F>,
    volume: &Array1<F>,
    fast_period: usize,
    slow_period: usize,
) -> Result<Array1<F>> {
    if high.len() != low.len() || low.len() != close.len() || close.len() != volume.len() {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: high.len(),
            actual: volume.len(),
        });
    }

    let n = high.len();
    let mut ad_line = Array1::zeros(n); // Accumulation/Distribution line
    let mut cumulative_ad = F::zero();

    // Calculate Accumulation/Distribution line
    for i in 0..n {
        let clv = if high[i] != low[i] {
            ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
        } else {
            F::zero()
        };

        let money_flow_volume = clv * volume[i];
        cumulative_ad = cumulative_ad + money_flow_volume;
        ad_line[i] = cumulative_ad;
    }

    // Calculate fast and slow EMAs of A/D line
    let fast_alpha = F::from(2.0).unwrap() / F::from(fast_period + 1).unwrap();
    let slow_alpha = F::from(2.0).unwrap() / F::from(slow_period + 1).unwrap();

    // Use basic EMA calculation
    use crate::financial::technical_indicators::basic::ema;
    let fast_ema = ema(&ad_line, fast_alpha)?;
    let slow_ema = ema(&ad_line, slow_alpha)?;

    // Chaikin Oscillator = Fast EMA - Slow EMA
    let chaikin_osc = &fast_ema - &slow_ema;

    Ok(chaikin_osc)
}

/// Fibonacci retracement levels structure
#[derive(Debug, Clone)]
pub struct FibonacciLevels<F: Float> {
    /// 100% Fibonacci level
    pub level_100: F,
    /// 78.6% Fibonacci level
    pub level_78_6: F,
    /// 61.8% Fibonacci level
    pub level_61_8: F,
    /// 50.0% Fibonacci level
    pub level_50_0: F,
    /// 38.2% Fibonacci level
    pub level_38_2: F,
    /// 23.6% Fibonacci level
    pub level_23_6: F,
    /// 0% Fibonacci level
    pub level_0: F,
}

/// Calculate Fibonacci Retracement Levels
///
/// Computes the standard Fibonacci retracement levels between a high and low price.
/// These levels are commonly used to identify potential support and resistance areas.
///
/// # Arguments
///
/// * `high_price` - The high price point
/// * `low_price` - The low price point
///
/// # Returns
///
/// * `Result<FibonacciLevels<F>>` - Fibonacci retracement levels
pub fn fibonacci_retracement<F: Float + Clone>(
    high_price: F,
    low_price: F,
) -> Result<FibonacciLevels<F>> {
    if high_price <= low_price {
        return Err(TimeSeriesError::InvalidInput(
            "High price must be greater than low price".to_string(),
        ));
    }

    let range = high_price - low_price;
    let fib_23_6 = F::from(0.236).unwrap();
    let fib_38_2 = F::from(0.382).unwrap();
    let fib_50_0 = F::from(0.5).unwrap();
    let fib_61_8 = F::from(0.618).unwrap();
    let fib_78_6 = F::from(0.786).unwrap();

    Ok(FibonacciLevels {
        level_100: high_price,
        level_78_6: high_price - range * fib_78_6,
        level_61_8: high_price - range * fib_61_8,
        level_50_0: high_price - range * fib_50_0,
        level_38_2: high_price - range * fib_38_2,
        level_23_6: high_price - range * fib_23_6,
        level_0: low_price,
    })
}

/// Kaufman's Adaptive Moving Average (KAMA)
///
/// An adaptive moving average that adjusts its smoothing constant based
/// on market volatility. More responsive in trending markets, less responsive
/// in sideways markets.
///
/// # Arguments
///
/// * `data` - Price data array
/// * `period` - Efficiency ratio calculation period
/// * `fast_sc` - Fast smoothing constant period
/// * `slow_sc` - Slow smoothing constant period
///
/// # Returns
///
/// * `Result<Array1<F>>` - KAMA values
pub fn kama<F: Float + Clone>(
    data: &Array1<F>,
    period: usize,
    fast_sc: usize,
    slow_sc: usize,
) -> Result<Array1<F>> {
    if data.len() < period {
        return Err(TimeSeriesError::InsufficientData {
            message: "Not enough data for KAMA calculation".to_string(),
            required: period,
            actual: data.len(),
        });
    }

    let n = data.len();
    let mut kama = Array1::zeros(n);
    kama[period - 1] = data[period - 1];

    let fast_alpha = F::from(2.0).unwrap() / F::from(fast_sc + 1).unwrap();
    let slow_alpha = F::from(2.0).unwrap() / F::from(slow_sc + 1).unwrap();

    for i in period..n {
        // Calculate efficiency ratio
        let direction = (data[i] - data[i - period]).abs();
        let volatility = (0..period)
            .map(|j| (data[i - j] - data[i - j - 1]).abs())
            .fold(F::zero(), |acc, x| acc + x);

        let efficiency_ratio = if volatility > F::zero() {
            direction / volatility
        } else {
            F::zero()
        };

        // Calculate smoothing constant
        let sc = efficiency_ratio * (fast_alpha - slow_alpha) + slow_alpha;
        let sc_squared = sc * sc;

        // Update KAMA
        kama[i] = kama[i - 1] + sc_squared * (data[i] - kama[i - 1]);
    }

    Ok(kama)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_bollinger_bands_basic() {
        let data = arr1(&[20.0, 21.0, 19.5, 22.0, 21.5, 20.0, 19.0, 23.0, 22.5, 21.0]);
        let config = BollingerBandsConfig {
            period: 5,
            std_dev_multiplier: 2.0,
            ma_type: MovingAverageType::Simple,
        };

        let result = bollinger_bands(&data, &config);
        assert!(result.is_ok());

        let bands = result.unwrap();
        assert_eq!(bands.upper_band.len(), data.len() - config.period + 1);

        // Upper band should be above middle, middle above lower
        for i in 0..bands.upper_band.len() {
            assert!(bands.upper_band[i] > bands.middle_band[i]);
            assert!(bands.middle_band[i] > bands.lower_band[i]);
        }
    }

    #[test]
    fn test_stochastic_oscillator() {
        // Use more data points for proper calculation with default config
        let high = arr1(&[
            15.0, 16.0, 14.5, 17.0, 16.5, 18.0, 17.5, 18.5, 19.0, 18.0, 17.0, 16.0, 17.5, 18.0,
            19.0,
        ]);
        let low = arr1(&[
            13.0, 14.0, 13.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 16.5, 15.5, 14.5, 16.0, 16.5,
            17.5,
        ]);
        let close = arr1(&[
            14.5, 15.5, 14.0, 16.0, 16.0, 17.0, 17.0, 18.0, 18.5, 17.5, 16.5, 15.5, 17.0, 17.5,
            18.5,
        ]);
        let config = StochasticConfig::default();

        let result = stochastic_oscillator(&high, &low, &close, &config);
        assert!(result.is_ok());

        let stoch = result.unwrap();

        // All values should be between 0 and 100
        for &value in stoch.percent_k.iter() {
            assert!(value >= 0.0 && value <= 100.0);
        }
        for &value in stoch.percent_d.iter() {
            assert!(value >= 0.0 && value <= 100.0);
        }
    }

    #[test]
    fn test_ichimoku_cloud() {
        let high = arr1(&[
            15.0, 16.0, 14.5, 17.0, 16.5, 18.0, 17.5, 19.0, 18.5, 20.0, 19.5, 21.0, 20.5, 22.0,
            21.5, 23.0, 22.5, 24.0, 23.5, 25.0, 24.5, 26.0, 25.5, 27.0, 26.5, 28.0, 27.5, 29.0,
            28.5, 30.0, 29.5, 31.0, 30.5, 32.0, 31.5, 33.0, 32.5, 34.0, 33.5, 35.0, 34.5, 36.0,
            35.5, 37.0, 36.5, 38.0, 37.5, 39.0, 38.5, 40.0, 39.5, 41.0, 40.5, 42.0,
        ]);
        let low = arr1(&[
            13.0, 14.0, 13.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0,
            20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0, 24.5, 25.0, 25.5, 26.0, 26.5, 27.0,
            27.5, 28.0, 28.5, 29.0, 29.5, 30.0, 30.5, 31.0, 31.5, 32.0, 32.5, 33.0, 33.5, 34.0,
            34.5, 35.0, 35.5, 36.0, 36.5, 37.0, 37.5, 38.0, 38.5, 39.0, 39.5, 40.0,
        ]);
        let close = arr1(&[
            14.5, 15.5, 14.0, 16.0, 16.0, 17.0, 17.0, 18.0, 18.0, 19.0, 19.0, 20.0, 20.0, 21.0,
            21.0, 22.0, 22.0, 23.0, 23.0, 24.0, 24.0, 25.0, 25.0, 26.0, 26.0, 27.0, 27.0, 28.0,
            28.0, 29.0, 29.0, 30.0, 30.0, 31.0, 31.0, 32.0, 32.0, 33.0, 33.0, 34.0, 34.0, 35.0,
            35.0, 36.0, 36.0, 37.0, 37.0, 38.0, 38.0, 39.0, 39.0, 40.0, 40.0, 41.0,
        ]);
        let config = IchimokuConfig::default();

        let result = ichimoku_cloud(&high, &low, &close, &config);
        assert!(result.is_ok());

        let cloud = result.unwrap();
        assert_eq!(cloud.tenkan_sen.len(), high.len());
        assert_eq!(cloud.kijun_sen.len(), high.len());
        assert_eq!(cloud.chikou_span.len(), high.len());
        assert_eq!(cloud.senkou_span_a.len(), high.len());
        assert_eq!(cloud.senkou_span_b.len(), high.len());
    }

    #[test]
    fn test_parabolic_sar() {
        let high = arr1(&[15.0, 16.0, 14.5, 17.0, 16.5, 18.0, 17.5, 19.0, 18.5]);
        let low = arr1(&[13.0, 14.0, 13.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5]);

        let result = parabolic_sar(&high, &low, 0.02, 0.2);
        assert!(result.is_ok());

        let sar = result.unwrap();
        assert_eq!(sar.len(), high.len());

        // All SAR values should be positive
        for &value in sar.iter() {
            assert!(value >= 0.0);
        }
    }

    #[test]
    fn test_fibonacci_retracement() {
        let result = fibonacci_retracement(100.0, 50.0);
        assert!(result.is_ok());

        let fib = result.unwrap();
        assert_eq!(fib.level_100, 100.0);
        assert_eq!(fib.level_0, 50.0);
        // Check that levels are properly calculated (between 0 and 100)
        assert!(fib.level_61_8 >= fib.level_0 && fib.level_61_8 <= fib.level_100);
        assert!(fib.level_50_0 >= fib.level_0 && fib.level_50_0 <= fib.level_100);
        assert!(fib.level_38_2 >= fib.level_0 && fib.level_38_2 <= fib.level_100);
    }

    #[test]
    fn test_kama() {
        let data = arr1(&[10.0, 12.0, 11.0, 13.0, 15.0, 14.0, 16.0, 18.0, 17.0, 19.0]);
        let result = kama(&data, 5, 2, 30);
        assert!(result.is_ok());

        let kama_values = result.unwrap();
        assert_eq!(kama_values.len(), data.len());

        // First period-1 values should be zero
        for i in 0..4 {
            assert_eq!(kama_values[i], 0.0);
        }

        // KAMA should be initialized at period-1 with data value
        assert_eq!(kama_values[4], data[4]);
    }

    #[test]
    fn test_insufficient_data() {
        let data = arr1(&[1.0, 2.0]);
        let config = BollingerBandsConfig {
            period: 5,
            std_dev_multiplier: 2.0,
            ma_type: MovingAverageType::Simple,
        };

        let result = bollinger_bands(&data, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_dimension_mismatch() {
        let high = arr1(&[15.0, 16.0, 14.5]);
        let low = arr1(&[13.0, 14.0]);
        let close = arr1(&[14.5, 15.5]);
        let config = StochasticConfig::default();

        let result = stochastic_oscillator(&high, &low, &close, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_adx_basic() {
        let high = arr1(&[
            15.0, 16.0, 14.5, 17.0, 16.5, 18.0, 17.5, 19.0, 18.5, 20.0, 19.5, 21.0, 20.5, 22.0,
            21.5,
        ]);
        let low = arr1(&[
            13.0, 14.0, 13.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0,
            20.5,
        ]);
        let close = arr1(&[
            14.5, 15.5, 14.0, 16.0, 16.0, 17.0, 17.0, 18.0, 18.0, 19.0, 19.0, 20.0, 20.0, 21.0,
            21.0,
        ]);

        let result = adx(&high, &low, &close, 5);
        assert!(result.is_ok());

        let adx_values = result.unwrap();

        // All ADX values should be between 0 and 100
        for &value in adx_values.iter() {
            assert!(value >= 0.0 && value <= 100.0);
        }
    }
}
