//! Comprehensive Financial Analysis and Modeling
//!
//! This module provides a complete suite of financial analysis tools for quantitative
//! finance, algorithmic trading, and risk management. It implements SciPy-compatible
//! APIs while leveraging Rust's performance and safety features for financial modeling.
//!
//! # Module Organization
//!
//! ## Volatility Models
//! [`models`] - Time series volatility models for financial data:
//! - GARCH family models (GARCH, EGARCH, GJR-GARCH, APARCH)
//! - Model selection, estimation, and forecasting
//! - Asymmetric and leverage effects modeling
//! - In-sample and out-of-sample evaluation
//!
//! ## Technical Analysis
//! [`technical_indicators`] - Comprehensive technical analysis indicators:
//! - **Basic indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, ATR
//! - **Advanced indicators**: Ichimoku, ADX, Parabolic SAR, VWAP, Stochastic
//! - **Volume indicators**: Chaikin Oscillator, MFI, OBV
//! - **Momentum indicators**: Williams %R, CCI, Aroon, KAMA
//! - Configurable parameters and multi-timeframe analysis
//!
//! ## Volatility Estimation
//! [`volatility`] - High-frequency volatility estimation techniques:
//! - **High-frequency estimators**: Garman-Klass, Rogers-Satchell, Yang-Zhang
//! - **Model-based estimators**: GARCH, EWMA (RiskMetrics)
//! - **Range-based estimators**: Parkinson, realized volatility
//! - Efficiency comparison and optimal estimator selection
//!
//! ## Risk Management
//! [`risk`] - Comprehensive risk analysis and measurement:
//! - **Value at Risk**: Historical, parametric, Monte Carlo methods
//! - **Expected Shortfall** (Conditional VaR) for tail risk
//! - **Risk-adjusted metrics**: Sharpe, Sortino, Information ratios
//! - **Drawdown analysis**: Maximum drawdown, recovery time, Pain Index
//! - Market risk measures (Beta, Alpha, Treynor ratio)
//!
//! ## Portfolio Management
//! [`portfolio`] - Modern portfolio theory and optimization:
//! - **Portfolio construction**: Mean-variance, risk parity, minimum variance
//! - **Performance analysis**: Comprehensive metrics and attribution
//! - **Risk decomposition**: Component VaR, factor analysis
//! - **Optimization algorithms**: Efficient frontier, maximum Sharpe ratio
//! - **Benchmarking**: Tracking error, information ratio, capture ratios
//!
//! ## Derivative Pricing
//! [`pricing`] - Financial instrument pricing models:
//! - **Black-Scholes**: European option pricing with Greeks
//! - **Implied volatility**: Market-based volatility extraction
//! - **Utility functions**: Present value, normal distributions
//! - Extensible framework for additional pricing models
//!
//! # Comprehensive Financial Analysis Workflow
//!
//! ## 1. Data Preparation and Preprocessing
//! ```rust,no_run
//! use scirs2_series::financial::{
//!     volatility::realized_volatility,
//!     technical_indicators::{sma, ema, rsi},
//! };
//! use ndarray::array;
//!
//! // Prepare price and return data (extend for RSI calculation)
//! let prices = array![100.0, 102.0, 101.5, 103.0, 102.5, 104.0, 105.5, 104.0, 106.0, 107.0,
//!                     106.5, 108.0, 107.5, 109.0, 108.5, 110.0, 111.5, 110.0, 112.0, 113.0];
//! let returns = array![0.02, -0.0049, 0.0148, -0.0049, 0.0146, 0.0144, -0.0142, 0.0192, 0.0094,
//!                      -0.0047, 0.0141, -0.0046, 0.0140, -0.0046, 0.0138, 0.0136, -0.0135, 0.0182, 0.0089];
//!
//! // Calculate basic technical indicators
//! let sma_5 = sma(&prices, 5).unwrap();
//! let ema_5 = ema(&prices, 0.2).unwrap();
//! let rsi_14 = rsi(&prices, 14).unwrap();
//!
//! // Estimate realized volatility
//! let vol = realized_volatility(&returns);
//! println!("Realized volatility: {:.4}", vol);
//! ```
//!
//! ## 2. Volatility Modeling and Forecasting
//! ```rust,no_run
//! use scirs2_series::financial::{
//!     models::{GarchModel, GarchConfig},
//!     volatility::{garman_klass_volatility, ewma_volatility},
//! };
//!
//! // Create larger returns dataset for GARCH fitting
//! let extended_returns = (0..100).map(|i| {
//!     0.01 * (i as f64 * 0.1).sin() + 0.005 * ((i as f64 * 0.05).cos() - 1.0)
//! }).collect::<Vec<f64>>();
//! let returns_array = ndarray::Array1::from_vec(extended_returns);
//!
//! // Fit GARCH model for volatility forecasting
//! let mut garch_model = GarchModel::new(GarchConfig::default());
//! let garch_result = garch_model.fit(&returns_array).unwrap();
//! println!("GARCH Log-likelihood: {:.2}", garch_result.log_likelihood);
//!
//! // Compare with high-frequency estimator (if OHLC data available)
//! // let high = array![...]; let low = array![...]; let close = array![...]; let open = array![...];
//! // let gk_vol = garman_klass_volatility(&high, &low, &close, &open).unwrap();
//!
//! // EWMA volatility (RiskMetrics approach)
//! let ewma_vol = ewma_volatility(&returns_array, 0.94).unwrap();
//! println!("EWMA volatility: {:.4}", ewma_vol);
//! ```
//!
//! ## 3. Risk Analysis and Management
//! ```rust,no_run
//! use scirs2_series::financial::risk::{
//!     var_historical, expected_shortfall, max_drawdown,
//!     sharpe_ratio, sortino_ratio, calmar_ratio,
//! };
//!
//! // Calculate VaR and Expected Shortfall (use extended returns)
//! let returns_array = ndarray::array![0.02, -0.0049, 0.0148, -0.0049, 0.0146, 0.0144];
//! let var_95 = var_historical(&returns_array, 0.95).unwrap();
//! let es_95 = expected_shortfall(&returns_array, 0.95).unwrap();
//! println!("95% VaR: {:.4}, 95% ES: {:.4}", var_95, es_95);
//!
//! // Risk-adjusted performance metrics
//! let risk_free_rate = 0.02;
//! let periods_per_year = 252;
//! let sharpe = sharpe_ratio(&returns_array, risk_free_rate, periods_per_year).unwrap();
//! let sortino = sortino_ratio(&returns_array, risk_free_rate, periods_per_year).unwrap();
//!
//! // Drawdown analysis (extend prices for drawdown calculation)
//! let extended_prices = (0..100).scan(100.0, |price, i| {
//!     let return_val = 0.01 * (i as f64 * 0.1).sin() + 0.005 * ((i as f64 * 0.05).cos() - 1.0);
//!     *price *= 1.0 + return_val;
//!     Some(*price)
//! }).collect::<Vec<f64>>();
//! let prices_array = ndarray::Array1::from_vec(extended_prices);
//! let max_dd = max_drawdown(&prices_array).unwrap();
//! let calmar = calmar_ratio(&returns_array, &prices_array, periods_per_year).unwrap();
//!
//! println!("Sharpe: {:.3}, Sortino: {:.3}, Calmar: {:.3}", sharpe, sortino, calmar);
//! println!("Max Drawdown: {:.2}%", max_dd * 100.0);
//! ```
//!
//! ## 4. Portfolio Construction and Optimization
//! ```rust,no_run
//! use scirs2_series::financial::portfolio::{
//!     Portfolio, risk_parity_portfolio, minimum_variance_portfolio,
//!     calculate_portfolio_returns, calculate_portfolio_metrics,
//! };
//! use ndarray::{Array2, array};
//!
//! // Multi-asset portfolio optimization
//! let asset_names = vec!["AAPL".to_string(), "GOOGL".to_string(), "MSFT".to_string()];
//!
//! // Prepare return matrix (rows: time, cols: assets)
//! // let asset_returns = Array2::from_shape_vec((252, 3), return_data).unwrap();
//!
//! // Risk parity portfolio
//! // let correlation_matrix = calculate_correlation_matrix(&asset_returns).unwrap();
//! // let risk_parity_weights = risk_parity_portfolio(&correlation_matrix).unwrap();
//! // let portfolio = Portfolio::new(risk_parity_weights, asset_names.clone()).unwrap();
//!
//! // Calculate portfolio performance
//! // let portfolio_returns = calculate_portfolio_returns(&asset_returns, portfolio.weights()).unwrap();
//! // let portfolio_metrics = calculate_portfolio_metrics(&portfolio_returns, &portfolio_prices, risk_free_rate, periods_per_year).unwrap();
//! ```
//!
//! ## 5. Derivative Pricing and Greeks Analysis
//! ```rust,no_run
//! use scirs2_series::financial::pricing::{
//!     black_scholes, black_scholes_greeks, implied_volatility,
//! };
//!
//! // Option pricing
//! let spot_price = 100.0;
//! let strike_price = 105.0;
//! let time_to_expiry = 0.25; // 3 months
//! let risk_free_rate = 0.05;
//! let volatility = 0.20;
//!
//! // Price call and put options
//! let call_price = black_scholes(spot_price, strike_price, time_to_expiry,
//!                               risk_free_rate, volatility, true).unwrap();
//! let put_price = black_scholes(spot_price, strike_price, time_to_expiry,
//!                              risk_free_rate, volatility, false).unwrap();
//!
//! // Calculate Greeks for risk management
//! let greeks = black_scholes_greeks(spot_price, strike_price, time_to_expiry,
//!                                  risk_free_rate, volatility, true).unwrap();
//!
//! println!("Call: ${:.2}, Put: ${:.2}", call_price, put_price);
//! println!("Delta: {:.4}, Gamma: {:.6}, Vega: {:.4}", greeks.delta, greeks.gamma, greeks.vega);
//!
//! // Implied volatility from market price
//! let market_price = 8.5;
//! if let Ok(impl_vol) = implied_volatility(market_price, spot_price, strike_price,
//!                                         time_to_expiry, risk_free_rate, true) {
//!     println!("Implied volatility: {:.2}%", impl_vol * 100.0);
//! }
//! ```
//!
//! ## 6. Advanced Technical Analysis
//! ```rust,no_run
//! use scirs2_series::financial::technical_indicators::{
//!     advanced_bollinger_bands, advanced_stochastic_oscillator, BollingerBandsConfig, StochasticConfig, IchimokuCloud,
//!     adx, parabolic_sar, vwap, chaikin_oscillator, MovingAverageType,
//! };
//!
//! // Advanced technical indicators (use extended prices)
//! let prices_array = ndarray::array![100.0, 101.0, 102.0, 101.5, 103.0, 102.5, 104.0, 103.5, 105.0, 104.5, 106.0];
//! let bb_conf = BollingerBandsConfig { period: 5, std_dev_multiplier: 2.0, ma_type: MovingAverageType::Simple };
//! let bb_result = advanced_bollinger_bands(&prices_array, &bb_conf).unwrap();
//! println!("Bollinger Bands - Upper: {:.2}, Lower: {:.2}", bb_result.upper_band[0], bb_result.lower_band[0]);
//!
//! // Momentum and trend indicators
//! let stoch_conf = StochasticConfig { k_period: 14, d_period: 3, d_smoothing: MovingAverageType::Simple };
//! // let stoch_result = advanced_stochastic_oscillator(&high, &low, &close, &stoch_conf).unwrap();
//!
//! // let adx_values = adx(&high, &low, &close, 14).unwrap();
//! // let sar_values = parabolic_sar(&high, &low, 0.02, 0.2).unwrap();
//! ```
//!
//! # Integration with SciRS2 Ecosystem
//!
//! The financial module seamlessly integrates with other SciRS2 modules:
//!
//! ## Time Series Analysis
//! ```rust,no_run
//! // Integration with scirs2-series for time series modeling
//! // Combined ARIMA-GARCH modeling for returns and volatility
//! ```
//!
//! ## Statistical Analysis  
//! ```rust,no_run
//! // Integration with scirs2-stats for hypothesis testing
//! // Statistical tests for model validation and backtesting
//! ```
//!
//! ## Optimization
//! ```rust,no_run
//! // Integration with scirs2-optimize for advanced portfolio optimization
//! // Constrained optimization with transaction costs and constraints
//! ```
//!
//! ## Machine Learning
//! ```rust,no_run
//! // Integration with scirs2-neural for predictive modeling
//! // Neural network-based volatility and return forecasting
//! ```
//!
//! # Performance and Scalability
//!
//! ## Computational Efficiency
//! - Optimized numerical algorithms with minimal allocations
//! - SIMD operations for vectorized calculations
//! - Parallel processing for portfolio and Monte Carlo operations
//! - Memory-efficient implementations for large datasets
//!
//! ## Production Ready
//! - Comprehensive error handling with detailed error messages
//! - Input validation and edge case handling
//! - Extensive test coverage with property-based testing
//! - Benchmarking against SciPy and industry standards
//!
//! # Industry Applications
//!
//! ## Asset Management
//! - Portfolio construction and rebalancing
//! - Risk budgeting and factor modeling
//! - Performance attribution and evaluation
//! - ESG integration and constraints
//!
//! ## Risk Management
//! - Market risk measurement (VaR, ES)
//! - Stress testing and scenario analysis  
//! - Credit risk modeling and correlation
//! - Operational risk quantification
//!
//! ## Trading and Market Making
//! - Real-time option pricing and Greeks
//! - Volatility surface construction
//! - Algorithmic trading signal generation
//! - Market microstructure analysis
//!
//! ## Regulatory Compliance
//! - Basel III capital requirements
//! - CCAR stress testing
//! - FRTB market risk modeling
//! - IFRS 17 insurance modeling
//!
//! # Future Development Roadmap
//!
//! ## Additional Models
//! - Jump diffusion models (Merton, Kou)
//! - Stochastic volatility models (Heston, SABR)
//! - Credit risk models (Merton, reduced form)
//! - Interest rate models (Vasicek, CIR, HJM)
//!
//! ## Advanced Pricing
//! - American option pricing (binomial, trinomial)
//! - Exotic options (barrier, Asian, lookback)
//! - Fixed income derivatives (bonds, swaps, caps/floors)
//! - Energy and commodity derivatives
//!
//! ## Machine Learning Integration
//! - Neural network volatility forecasting
//! - Reinforcement learning for portfolio optimization
//! - Alternative data integration
//! - ESG factor modeling

pub mod models;
pub mod portfolio;
pub mod pricing;
pub mod risk;
pub mod technical_indicators;
pub mod volatility;

// Re-export the most commonly used types and functions for convenience

// Volatility models
pub use models::{
    AparchModel, AparchResult, EgarchConfig, EgarchModel, EgarchResult, GarchConfig, GarchModel,
    GarchResult, GjrGarchModel, GjrGarchResult,
};

// Technical indicators (basic)
pub use technical_indicators::{
    atr, bollinger_bands, cci, ema, macd, obv, rsi, sma, stochastic, williams_r,
};

// Technical indicators (advanced)
pub use technical_indicators::{
    adx, aroon, chaikin_oscillator, fibonacci_retracement, kama, mfi, parabolic_sar, vwap,
    AdvancedBollingerBands, AdvancedStochasticOscillator, IchimokuCloud,
};

// Volatility estimators
pub use volatility::{
    ewma_volatility, garch_volatility_estimate, garman_klass_volatility, intraday_volatility,
    parkinson_volatility, range_volatility, realized_volatility, rogers_satchell_volatility,
    yang_zhang_volatility,
};

// Risk management
pub use risk::{
    // Market risk measures
    beta,
    calculate_drawdown_series,
    calmar_ratio,
    drawdown_recovery_analysis,
    expected_shortfall,
    information_ratio,
    jensens_alpha,
    // Drawdown analysis
    max_drawdown,
    monte_carlo_var,
    omega_ratio,
    pain_index,
    parametric_var,
    // Risk-adjusted ratios
    sharpe_ratio,
    sortino_ratio,
    treynor_ratio,
    ulcer_index,
    // VaR and Expected Shortfall
    var_historical,
    DrawdownPeriod,
};

// Portfolio management
pub use portfolio::{
    analyze_correlation_risk,
    calculate_capture_ratios,
    calculate_component_var,
    calculate_correlation_matrix,
    // Optimization
    calculate_efficient_portfolio,
    calculate_information_ratio,
    calculate_portfolio_beta,
    // Performance analysis
    calculate_portfolio_metrics,
    calculate_portfolio_returns,
    calculate_portfolio_variance,
    calculate_return_statistics,
    calculate_tracking_error,
    maximum_sharpe_portfolio,
    minimum_variance_portfolio,
    // Risk analysis
    portfolio_var_parametric,
    risk_parity_portfolio,
    stress_test_portfolio,
    // Core structures
    Portfolio,
    PortfolioMetrics,
};

// Pricing models
pub use pricing::{
    // Option pricing
    black_scholes,
    black_scholes_greeks,
    calculate_d1,
    calculate_d2,
    future_value,
    implied_volatility,
    // Utility functions
    normal_cdf,
    normal_pdf,
    normal_quantile,
    option_value_components,
    present_value,
    Greeks,
};
