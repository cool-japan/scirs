//! Risk Management and Analysis
//!
//! This module provides comprehensive risk management tools for financial
//! analysis, including Value at Risk (VaR), risk-adjusted performance metrics,
//! drawdown analysis, and portfolio risk assessment.
//!
//! # Module Organization
//!
//! ## Core Metrics
//! [`metrics`] - Fundamental risk measurement techniques:
//! - Value at Risk (Historical, Parametric, Monte Carlo)
//! - Expected Shortfall (Conditional VaR)
//! - Risk-adjusted performance ratios
//! - Market risk measures (Beta, Alpha)
//!
//! ## Drawdown Analysis
//! [`drawdown`] - Comprehensive drawdown analysis tools:
//! - Maximum drawdown calculation
//! - Pain Index and Ulcer Index
//! - Recovery time analysis
//! - Calmar Ratio calculation
//!
//! # Risk Management Framework
//!
//! ## Risk Measurement Hierarchy
//!
//! 1. **Basic Risk Metrics**
//!    - Standard deviation (total risk)
//!    - Downside deviation (harmful volatility)
//!    - Beta (systematic risk)
//!
//! 2. **Advanced Risk Measures**
//!    - Value at Risk (tail risk)
//!    - Expected Shortfall (extreme tail risk)
//!    - Maximum drawdown (worst-case scenarios)
//!
//! 3. **Risk-Adjusted Performance**
//!    - Sharpe Ratio (total risk-adjusted)
//!    - Sortino Ratio (downside risk-adjusted)
//!    - Information Ratio (active risk-adjusted)
//!    - Treynor Ratio (systematic risk-adjusted)
//!
//! ## Use Cases by Risk Type
//!
//! ### Market Risk
//! Use VaR, Beta, and correlation analysis to measure sensitivity
//! to market movements.
//!
//! ### Credit Risk
//! Apply Expected Shortfall and drawdown analysis to assess
//! potential default losses.
//!
//! ### Operational Risk
//! Use Maximum Drawdown and recovery analysis for operational
//! failure scenarios.
//!
//! ### Liquidity Risk
//! Employ drawdown duration and recovery time analysis.
//!
//! # Examples
//!
//! ## Comprehensive Risk Assessment
//! ```rust
//! use scirs2_series::financial::risk::{metrics, drawdown};
//! use scirs2_core::ndarray::array;
//!
//! // Portfolio data
//! let returns = array![0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.003, 0.007];
//! let portfolio_values = array![1000.0, 1010.0, 990.0, 1005.0, 997.0, 1009.0, 1014.0, 1011.0, 1018.0];
//! let market_returns = array![0.008, -0.018, 0.012, -0.006, 0.010, 0.004, -0.002, 0.006];
//!
//! // Risk metrics
//! let var_95 = metrics::var_historical(&returns, 0.95).unwrap();
//! let es_95 = metrics::expected_shortfall(&returns, 0.95).unwrap();
//! let sharpe = metrics::sharpe_ratio(&returns, 0.02, 252).unwrap();
//! let beta = metrics::beta(&returns, &market_returns).unwrap();
//!
//! // Drawdown analysis
//! let max_dd = drawdown::max_drawdown(&portfolio_values).unwrap();
//! let pain_idx = drawdown::pain_index(&portfolio_values).unwrap();
//! let ulcer_idx = drawdown::ulcer_index(&portfolio_values).unwrap();
//!
//! println!("Risk Assessment:");
//! println!("VaR (95%): {:.4}", var_95);
//! println!("Expected Shortfall: {:.4}", es_95);
//! println!("Sharpe Ratio: {:.4}", sharpe);
//! println!("Beta: {:.4}", beta);
//! println!("Max Drawdown: {:.4}", max_dd);
//! println!("Pain Index: {:.4}", pain_idx);
//! println!("Ulcer Index: {:.4}", ulcer_idx);
//! ```
//!
//! ## Risk-Adjusted Performance Comparison
//! ```rust
//! use scirs2_series::financial::risk::metrics::{sharpe_ratio, sortino_ratio, information_ratio};
//! use scirs2_core::ndarray::array;
//!
//! let portfolio_a = array![0.012, -0.015, 0.018, -0.005, 0.010];
//! let portfolio_b = array![0.008, -0.012, 0.015, -0.008, 0.012];
//! let benchmark = array![0.010, -0.018, 0.016, -0.006, 0.011];
//! let risk_free = 0.02;
//!
//! // Compare risk-adjusted performance
//! let sharpe_a = sharpe_ratio(&portfolio_a, risk_free, 252).unwrap();
//! let sharpe_b = sharpe_ratio(&portfolio_b, risk_free, 252).unwrap();
//!
//! let sortino_a = sortino_ratio(&portfolio_a, risk_free, 252).unwrap();
//! let sortino_b = sortino_ratio(&portfolio_b, risk_free, 252).unwrap();
//!
//! let info_a = information_ratio(&portfolio_a, &benchmark).unwrap();
//! let info_b = information_ratio(&portfolio_b, &benchmark).unwrap();
//!
//! println!("Portfolio A - Sharpe: {:.3}, Sortino: {:.3}, Info: {:.3}", sharpe_a, sortino_a, info_a);
//! println!("Portfolio B - Sharpe: {:.3}, Sortino: {:.3}, Info: {:.3}", sharpe_b, sortino_b, info_b);
//! ```
//!
//! ## Drawdown Recovery Analysis
//! ```rust
//! use scirs2_series::financial::risk::drawdown::{
//!     drawdown_recovery_analysis, max_consecutive_losses, average_recovery_time
//! };
//! use scirs2_core::ndarray::array;
//!
//! let portfolio_values = array![1000.0, 1100.0, 1050.0, 950.0, 1200.0, 1150.0, 1300.0, 1250.0, 1400.0];
//! let returns = array![0.10, -0.045, -0.095, 0.263, -0.042, 0.130, -0.038, 0.120];
//!
//! // Detailed drawdown analysis
//! let dd_periods = drawdown_recovery_analysis(&portfolio_values).unwrap();
//! let max_losses = max_consecutive_losses(&returns);
//! let avg_recovery = average_recovery_time(&portfolio_values).unwrap();
//!
//! println!("Drawdown Analysis:");
//! println!("Number of drawdown periods: {}", dd_periods.len());
//! println!("Max consecutive losses: {}", max_losses);
//! println!("Average recovery time: {:.1} periods", avg_recovery);
//!
//! for (i, period) in dd_periods.iter().enumerate() {
//!     println!("Period {}: {:.2}% drawdown, {} periods duration",
//!              i + 1, period.max_drawdown * 100.0, period.duration);
//! }
//! ```
//!
//! # Risk Management Best Practices
//!
//! ## VaR Modeling
//!
//! 1. **Use multiple VaR methods** for robustness
//! 2. **Backtest VaR models** regularly
//! 3. **Complement VaR with ES** for tail risk
//! 4. **Consider time-varying volatility**
//!
//! ## Drawdown Analysis
//!
//! 1. **Monitor maximum drawdown** for position sizing
//! 2. **Use recovery analysis** for risk budgeting
//! 3. **Consider correlation** during stress periods
//! 4. **Implement stop-loss rules** based on drawdown limits
//!
//! ## Performance Evaluation
//!
//! 1. **Use multiple risk-adjusted ratios** for comparison
//! 2. **Consider benchmark-relative metrics**
//! 3. **Analyze across different time periods**
//! 4. **Account for changing risk characteristics**
//!
//! # Implementation Notes
//!
//! ## Numerical Considerations
//!
//! All risk calculations are designed to handle:
//! - Extreme values and outliers
//! - Missing data points
//! - Numerical precision issues
//! - Edge cases (zero variance, infinite ratios)
//!
//! ## Performance Optimization
//!
//! The implementations use:
//! - Efficient sorting algorithms for quantile calculations
//! - Vectorized operations where possible
//! - Memory-efficient calculations for large datasets
//! - Caching of intermediate results
//!
//! ## Regulatory Compliance
//!
//! Risk metrics follow industry standards:
//! - Basel III for VaR and Expected Shortfall
//! - GIPS standards for performance measurement
//! - CFA Institute guidelines for risk-adjusted returns

pub mod drawdown;
pub mod metrics;

// Re-export commonly used functions for convenience
pub use metrics::{
    beta, expected_shortfall, information_ratio, jensens_alpha, monte_carlo_var, omega_ratio,
    parametric_var, sharpe_ratio, sortino_ratio, treynor_ratio, var_historical,
};

pub use drawdown::{
    average_drawdown_duration, average_recovery_time, calculate_drawdown_series, calmar_ratio,
    drawdown_recovery_analysis, max_consecutive_losses, max_drawdown, pain_index, ulcer_index,
    DrawdownPeriod,
};
