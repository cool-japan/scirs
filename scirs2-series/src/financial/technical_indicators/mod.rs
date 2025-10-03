//! Technical Indicators for Financial Analysis
//!
//! This module provides a comprehensive collection of technical indicators
//! used in financial market analysis. The indicators are organized into
//! basic and advanced categories based on their complexity and functionality.
//!
//! # Module Organization
//!
//! ## Basic Technical Indicators
//! [`basic`] - Fundamental indicators commonly used in technical analysis:
//! - Moving averages (SMA, EMA)
//! - Momentum oscillators (RSI, MACD)
//! - Volatility indicators (ATR, Bollinger Bands)
//! - Volume indicators (OBV)
//! - Price-based oscillators (Stochastic, Williams %R, CCI)
//!
//! ## Advanced Technical Indicators  
//! [`advanced`] - Sophisticated indicators for complex analysis:
//! - Multi-dimensional trend analysis (Ichimoku Cloud)
//! - Adaptive indicators (KAMA)
//! - Advanced volume-price analysis (VWAP, Chaikin Oscillator, MFI)
//! - Trend strength indicators (ADX, Parabolic SAR)
//! - Market structure analysis (Aroon, Fibonacci levels)
//! - Enhanced oscillators with configuration options
//!
//! # Usage Guidelines
//!
//! ## Choosing the Right Indicator
//!
//! - **Trend Analysis**: Use SMA/EMA (basic) or Ichimoku/ADX (advanced)
//! - **Momentum**: Use RSI/MACD (basic) or enhanced Stochastic (advanced)
//! - **Volatility**: Use ATR (basic) or enhanced Bollinger Bands (advanced)
//! - **Volume Analysis**: Use OBV (basic) or VWAP/MFI (advanced)
//! - **Support/Resistance**: Use basic Bollinger Bands or Fibonacci levels
//!
//! ## Performance Considerations
//!
//! Basic indicators are optimized for speed and simplicity, while advanced
//! indicators provide more features but with increased computational cost.
//!
//! # Examples
//!
//! ## Using Basic Indicators
//! ```rust
//! use scirs2_series::financial::technical_indicators::basic::{sma, rsi, bollinger_bands};
//! use scirs2_core::ndarray::array;
//!
//! let prices = array![10.0, 11.0, 12.0, 11.5, 13.0, 14.0, 13.5, 15.0];
//!
//! // Simple moving average
//! let sma_values = sma(&prices, 3).unwrap();
//!
//! // RSI momentum oscillator
//! let rsi_values = rsi(&prices, 6).unwrap();
//!
//! // Basic Bollinger Bands
//! let (upper, middle, lower) = bollinger_bands(&prices, 5, 2.0).unwrap();
//! ```
//!
//! ## Using Advanced Indicators
//! ```rust
//! use scirs2_series::financial::technical_indicators::advanced::{
//!     BollingerBandsConfig, MovingAverageType, bollinger_bands, kama
//! };
//! use scirs2_core::ndarray::array;
//!
//! let prices = array![20.0, 21.0, 19.5, 22.0, 21.5, 20.0, 19.0, 23.0, 22.5, 21.0];
//!
//! // Enhanced Bollinger Bands with configuration
//! let config = BollingerBandsConfig {
//!     period: 5,
//!     std_dev_multiplier: 2.0,
//!     ma_type: MovingAverageType::Exponential,
//! };
//! let bands = bollinger_bands(&prices, &config).unwrap();
//!
//! // Adaptive moving average
//! let kama_values = kama(&prices, 5, 2, 30).unwrap();
//! ```
//!
//! ## Combining Multiple Indicators
//! ```rust
//! use scirs2_series::financial::technical_indicators::{basic, advanced};
//! use scirs2_core::ndarray::array;
//!
//! let prices = array![10.0, 12.0, 11.0, 13.0, 15.0, 14.0, 16.0, 18.0];
//!
//! // Trend analysis with multiple indicators
//! let sma_trend = basic::sma(&prices, 3).unwrap();
//! let kama_adaptive = advanced::kama(&prices, 5, 2, 30).unwrap();
//!
//! // Momentum confirmation
//! let rsi_momentum = basic::rsi(&prices, 6).unwrap();
//! ```

pub mod advanced;
pub mod basic;

// Re-export commonly used types and functions for convenience
pub use basic::{atr, bollinger_bands, cci, ema, macd, obv, rsi, sma, stochastic, williams_r};

pub use advanced::{
    adx, aroon, bollinger_bands as advanced_bollinger_bands, chaikin_oscillator,
    fibonacci_retracement, ichimoku_cloud, kama, mfi, parabolic_sar,
    stochastic_oscillator as advanced_stochastic_oscillator, vwap,
    BollingerBands as AdvancedBollingerBands, BollingerBandsConfig, FibonacciLevels, IchimokuCloud,
    IchimokuConfig, MovingAverageType, StochasticConfig,
    StochasticOscillator as AdvancedStochasticOscillator,
};
