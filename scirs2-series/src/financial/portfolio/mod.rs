//! Portfolio Management and Analysis
//!
//! This module provides comprehensive portfolio management tools including
//! portfolio construction, performance analysis, optimization, and risk management.
//! It supports modern portfolio theory, risk parity, and various other optimization
//! approaches commonly used in quantitative finance.
//!
//! # Module Organization
//!
//! ## Core Structures
//! [`core`] - Fundamental portfolio data structures:
//! - `Portfolio` - Portfolio weights and metadata management
//! - `PortfolioMetrics` - Comprehensive performance measurement
//! - Basic portfolio operations and validation
//!
//! ## Performance Analysis
//! [`metrics`] - Portfolio performance measurement tools:
//! - Comprehensive performance metrics calculation
//! - Risk-adjusted performance ratios
//! - Rolling performance analysis
//! - Benchmark-relative metrics
//!
//! ## Portfolio Optimization
//! [`optimization`] - Modern portfolio optimization algorithms:
//! - Mean-variance optimization (efficient frontier)
//! - Risk parity and minimum variance portfolios
//! - Maximum Sharpe ratio optimization
//! - Correlation matrix utilities
//!
//! ## Risk Management
//! [`risk`] - Portfolio-specific risk analysis:
//! - Portfolio VaR and Expected Shortfall
//! - Component risk decomposition
//! - Stress testing and scenario analysis
//! - Correlation-based risk measures
//!
//! # Portfolio Construction Workflow
//!
//! ## 1. Data Preparation
//! ```rust
//! use scirs2_series::financial::portfolio::{core::Portfolio, optimization::calculate_correlation_matrix};
//! use scirs2_core::ndarray::{Array1, Array2, array};
//!
//! // Prepare asset return data (generate synthetic returns)
//! let returns_data: Vec<f64> = (0..756).map(|i| {
//!     let asset = i % 3;
//!     let day = i / 3;
//!     0.001 * (day as f64 * 0.01 + asset as f64).sin() + 0.0005 * ((day as f64 * 0.02).cos() - 1.0)
//! }).collect();
//! let returns = Array2::from_shape_vec((252, 3), returns_data).unwrap();
//! let asset_names = vec!["AAPL".to_string(), "GOOGL".to_string(), "MSFT".to_string()];
//!
//! // Calculate correlation matrix for optimization
//! let correlation_matrix = calculate_correlation_matrix(&returns).unwrap();
//! ```
//!
//! ## 2. Portfolio Optimization
//! ```rust
//! use scirs2_series::financial::portfolio::optimization::{
//!     risk_parity_portfolio, minimum_variance_portfolio, calculate_efficient_portfolio,
//!     calculate_correlation_matrix
//! };
//! use scirs2_core::ndarray::{Array1, Array2, array};
//!
//! // Setup returns data (needed for this doctest block)
//! let returns_data: Vec<f64> = (0..756).map(|i| {
//!     let asset = i % 3;
//!     let day = i / 3;
//!     0.001 * (day as f64 * 0.01 + asset as f64).sin() + 0.0005 * ((day as f64 * 0.02).cos() - 1.0)
//! }).collect();
//! let returns = Array2::from_shape_vec((252, 3), returns_data).unwrap();
//! let correlation_matrix = calculate_correlation_matrix(&returns).unwrap();
//!
//! // Convert correlation to covariance (simplified example)
//! let volatilities = array![0.15, 0.18, 0.12]; // Asset volatilities
//! let mut covariance = correlation_matrix.clone();
//! for i in 0..3 {
//!     for j in 0..3 {
//!         covariance[[i, j]] *= volatilities[i] * volatilities[j];
//!     }
//! }
//!
//! // Different optimization approaches
//! let risk_parity_weights = risk_parity_portfolio(&covariance).unwrap();
//! let min_var_weights = minimum_variance_portfolio(&covariance).unwrap();
//!
//! // Create efficient frontier portfolio
//! let expected_returns = array![0.10, 0.12, 0.08];
//! let target_return = 0.10;
//! let efficient_weights = calculate_efficient_portfolio(&expected_returns, &covariance, target_return).unwrap();
//! ```
//!
//! ## 3. Portfolio Construction
//! ```rust
//! use scirs2_series::financial::portfolio::{core::Portfolio, optimization::{risk_parity_portfolio, calculate_correlation_matrix}};
//! use scirs2_core::ndarray::{Array2, array};
//!
//! // Setup data for this doctest block
//! let returns_data: Vec<f64> = (0..756).map(|i| {
//!     let asset = i % 3;
//!     let day = i / 3;
//!     0.001 * (day as f64 * 0.01 + asset as f64).sin() + 0.0005 * ((day as f64 * 0.02).cos() - 1.0)
//! }).collect();
//! let returns = Array2::from_shape_vec((252, 3), returns_data).unwrap();
//! let correlation_matrix = calculate_correlation_matrix(&returns).unwrap();
//! let volatilities = array![0.15, 0.18, 0.12];
//! let mut covariance = correlation_matrix.clone();
//! for i in 0..3 {
//!     for j in 0..3 {
//!         covariance[[i, j]] *= volatilities[i] * volatilities[j];
//!     }
//! }
//! let risk_parity_weights = risk_parity_portfolio(&covariance).unwrap();
//! let asset_names = vec!["AAPL".to_string(), "GOOGL".to_string(), "MSFT".to_string()];
//!
//! // Create portfolio with optimized weights
//! let portfolio: Portfolio<f64> = Portfolio::new(risk_parity_weights, asset_names.clone()).unwrap();
//!
//! // Or create equal-weight portfolio for comparison
//! let equal_weight_portfolio: Portfolio<f64> = Portfolio::equal_weight(3, asset_names).unwrap();
//! ```
//!
//! ## 4. Performance Analysis
//! ```rust
//! use scirs2_series::financial::portfolio::{
//!     core::{Portfolio, calculate_portfolio_returns},
//!     metrics::calculate_portfolio_metrics,
//!     optimization::{risk_parity_portfolio, calculate_correlation_matrix}
//! };
//! use scirs2_core::ndarray::{Array1, Array2, array};
//!
//! // Setup data for this doctest block
//! let returns_data: Vec<f64> = (0..756).map(|i| {
//!     let asset = i % 3;
//!     let day = i / 3;
//!     0.001 * (day as f64 * 0.01 + asset as f64).sin() + 0.0005 * ((day as f64 * 0.02).cos() - 1.0)
//! }).collect();
//! let returns = Array2::from_shape_vec((252, 3), returns_data).unwrap();
//! let correlation_matrix = calculate_correlation_matrix(&returns).unwrap();
//! let volatilities = array![0.15, 0.18, 0.12];
//! let mut covariance = correlation_matrix.clone();
//! for i in 0..3 {
//!     for j in 0..3 {
//!         covariance[[i, j]] *= volatilities[i] * volatilities[j];
//!     }
//! }
//! let risk_parity_weights = risk_parity_portfolio(&covariance).unwrap();
//! let asset_names = vec!["AAPL".to_string(), "GOOGL".to_string(), "MSFT".to_string()];
//! let portfolio = Portfolio::new(risk_parity_weights, asset_names).unwrap();
//!
//! // Calculate portfolio returns
//! let portfolio_returns = calculate_portfolio_returns(&returns, portfolio.weights()).unwrap();
//!
//! // Convert returns to prices for drawdown analysis
//! let mut portfolio_prices = Array1::zeros(portfolio_returns.len() + 1);
//! portfolio_prices[0] = 1000.0; // Starting value
//! for i in 0..portfolio_returns.len() {
//!     portfolio_prices[i + 1] = portfolio_prices[i] * (1.0 + portfolio_returns[i]);
//! }
//!
//! // Calculate comprehensive performance metrics
//! let risk_free_rate = 0.02; // 2% annual
//! let periods_per_year = 252; // Daily data
//! let metrics = calculate_portfolio_metrics(
//!     &portfolio_returns, &portfolio_prices, risk_free_rate, periods_per_year
//! ).unwrap();
//!
//! println!("Sharpe Ratio: {:.3}", metrics.sharpe_ratio);
//! println!("Max Drawdown: {:.2}%", metrics.max_drawdown * 100.0);
//! println!("VaR (95%): {:.2}%", metrics.var_95 * 100.0);
//! ```
//!
//! ## 5. Risk Analysis
//! ```rust
//! use scirs2_series::financial::portfolio::{
//!     risk::{portfolio_var_parametric, calculate_component_var, stress_test_portfolio},
//!     core::Portfolio,
//!     optimization::{risk_parity_portfolio, calculate_correlation_matrix}
//! };
//! use scirs2_core::ndarray::{Array2, array};
//!
//! // Setup data for this doctest block
//! let returns_data: Vec<f64> = (0..756).map(|i| {
//!     let asset = i % 3;
//!     let day = i / 3;
//!     0.001 * (day as f64 * 0.01 + asset as f64).sin() + 0.0005 * ((day as f64 * 0.02).cos() - 1.0)
//! }).collect();
//! let returns = Array2::from_shape_vec((252, 3), returns_data).unwrap();
//! let correlation_matrix = calculate_correlation_matrix(&returns).unwrap();
//! let volatilities = array![0.15, 0.18, 0.12];
//! let mut covariance = correlation_matrix.clone();
//! for i in 0..3 {
//!     for j in 0..3 {
//!         covariance[[i, j]] *= volatilities[i] * volatilities[j];
//!     }
//! }
//! let risk_parity_weights = risk_parity_portfolio(&covariance).unwrap();
//! let asset_names = vec!["AAPL".to_string(), "GOOGL".to_string(), "MSFT".to_string()];
//! let portfolio = Portfolio::new(risk_parity_weights, asset_names).unwrap();
//!
//! // Parametric VaR for specific dollar amount
//! let portfolio_value = 1_000_000.0;
//! let mean_return = 0.0008; // Daily
//! let return_std = 0.015;   // Daily
//! let var_dollar = portfolio_var_parametric(portfolio_value, mean_return, return_std, 0.95, 1).unwrap();
//! println!("1-day 95% VaR: ${:.2}", var_dollar);
//!
//! // Component VaR decomposition
//! let component_vars = calculate_component_var(portfolio.weights(), &returns, 0.95).unwrap();
//! for (i, &comp_var) in component_vars.iter().enumerate() {
//!     println!("Asset {} Component VaR: {:.4}", i, comp_var);
//! }
//!
//! // Stress testing
//! let stress_factors = array![1.5, 2.0, 3.0]; // 1.5x, 2x, 3x worst case
//! let stress_results = stress_test_portfolio(portfolio.weights(), &returns, &stress_factors).unwrap();
//! ```
//!
//! # Optimization Approaches Comparison
//!
//! | Method | Objective | Best For | Limitations |
//! |--------|-----------|----------|-------------|
//! | **Equal Weight** | Simplicity | Diversification, low maintenance | Ignores risk differences |
//! | **Mean-Variance** | Maximize return/risk ratio | Clear return forecasts | Sensitive to input errors |
//! | **Risk Parity** | Equal risk contribution | Stable, risk-balanced | May ignore returns |
//! | **Minimum Variance** | Minimize portfolio risk | Conservative allocation | May sacrifice returns |
//! | **Maximum Sharpe** | Maximize risk-adjusted return | Optimal risk/return | Requires accurate forecasts |
//!
//! # Risk Management Framework
//!
//! ## Portfolio Risk Hierarchy
//!
//! 1. **Total Risk**
//!    - Portfolio volatility (standard deviation)
//!    - Value at Risk (VaR) at various confidence levels
//!    - Expected Shortfall (Conditional VaR)
//!
//! 2. **Component Risk**
//!    - Asset contribution to portfolio VaR
//!    - Marginal risk contribution
//!    - Risk decomposition by sector/factor
//!
//! 3. **Scenario Risk**
//!    - Stress test results
//!    - Historical scenario replays
//!    - Monte Carlo simulation results
//!
//! ## Performance Measurement
//!
//! ### Risk-Adjusted Returns
//! - **Sharpe Ratio**: Total risk-adjusted performance
//! - **Sortino Ratio**: Downside risk-adjusted performance  
//! - **Calmar Ratio**: Return/Maximum Drawdown ratio
//! - **Information Ratio**: Active return/Tracking Error
//!
//! ### Benchmark Analysis
//! - **Beta**: Systematic risk relative to benchmark
//! - **Alpha**: Risk-adjusted excess return (Jensen's Alpha)
//! - **Tracking Error**: Standard deviation of active returns
//! - **Up/Down Capture**: Performance in rising/falling markets
//!
//! # Implementation Examples
//!
//! ## Multi-Asset Portfolio Construction
//! ```rust
//! use scirs2_series::financial::portfolio::{core, optimization, metrics};
//! use scirs2_core::ndarray::{Array1, Array2, array};
//!
//! // Define asset universe
//! let asset_names = vec![
//!     "US_Equity".to_string(), "Intl_Equity".to_string(),
//!     "Bonds".to_string(), "Commodities".to_string()
//! ];
//!
//! // Historical returns matrix (rows: time, cols: assets) - synthetic data
//! let returns_data: Vec<f64> = (0..1008).map(|i| {
//!     let asset = i % 4;
//!     let day = i / 4;
//!     0.0008 * (day as f64 * 0.01 + asset as f64).sin() + 0.0003 * ((day as f64 * 0.02).cos() - 1.0)
//! }).collect();
//! let returns_data = Array2::from_shape_vec((252, 4), returns_data).unwrap();
//!
//! // Calculate optimization inputs
//! let correlation_matrix = optimization::calculate_correlation_matrix(&returns_data).unwrap();
//! let expected_returns = array![0.08, 0.09, 0.04, 0.06]; // Annual expected returns
//!
//! // Build covariance matrix
//! let volatilities = array![0.16, 0.18, 0.05, 0.22]; // Annual volatilities
//! let mut covariance_matrix = correlation_matrix;
//! for i in 0..4 {
//!     for j in 0..4 {
//!         covariance_matrix[[i, j]] *= volatilities[i] * volatilities[j];
//!     }
//! }
//!
//! // Compare different allocation strategies
//! let strategies: [(&str, core::Portfolio<f64>); 3] = [
//!     ("Equal Weight", core::Portfolio::equal_weight(4, asset_names.clone()).unwrap()),
//!     ("Risk Parity", core::Portfolio::new(
//!         optimization::risk_parity_portfolio(&covariance_matrix).unwrap(),
//!         asset_names.clone()
//!     ).unwrap()),
//!     ("Min Variance", core::Portfolio::new(
//!         optimization::minimum_variance_portfolio(&covariance_matrix).unwrap(),
//!         asset_names.clone()
//!     ).unwrap()),
//! ];
//!
//! // Analyze each strategy
//! for (name, portfolio) in strategies.iter() {
//!     let portfolio_returns = core::calculate_portfolio_returns(&returns_data, portfolio.weights()).unwrap();
//!     
//!     // Convert to prices for analysis
//!     let mut prices = Array1::zeros(portfolio_returns.len() + 1);
//!     prices[0] = 1000.0;
//!     for i in 0..portfolio_returns.len() {
//!         prices[i + 1] = prices[i] * (1.0 + portfolio_returns[i]);
//!     }
//!     
//!     let metrics = metrics::calculate_portfolio_metrics(&portfolio_returns, &prices, 0.025, 252).unwrap();
//!     
//!     println!("{}: Sharpe {:.3}, MaxDD {:.2}%", name, metrics.sharpe_ratio, metrics.max_drawdown * 100.0);
//! }
//! ```
//!
//! # Best Practices
//!
//! ## Data Quality
//! - Use sufficient historical data (minimum 2-3 years)
//! - Handle missing data and outliers appropriately
//! - Consider regime changes and structural breaks
//! - Update correlation/covariance estimates regularly
//!
//! ## Optimization Robustness
//! - Use multiple optimization methods for comparison
//! - Implement constraints (max weight, sector limits)
//! - Consider transaction costs and turnover
//! - Regularly rebalance based on strategy requirements
//!
//! ## Risk Management
//! - Monitor multiple risk measures (VaR, ES, drawdown)
//! - Perform regular stress testing
//! - Decompose risk by component and factor
//! - Set risk limits and monitoring triggers
//!
//! ## Performance Analysis
//! - Compare against relevant benchmarks
//! - Analyze performance across different market regimes
//! - Consider both absolute and risk-adjusted metrics
//! - Account for costs, taxes, and implementation constraints

pub mod core;
pub mod metrics;
pub mod optimization;
pub mod risk;

// Re-export commonly used types and functions for convenience
pub use core::{calculate_portfolio_returns, Portfolio, PortfolioMetrics};

pub use metrics::{
    calculate_capture_ratios, calculate_information_ratio, calculate_portfolio_beta,
    calculate_portfolio_metrics, calculate_return_statistics, calculate_rolling_metrics,
    calculate_tracking_error,
};

pub use optimization::{
    calculate_correlation_matrix, calculate_efficient_portfolio, calculate_portfolio_variance,
    maximum_diversification_portfolio, maximum_sharpe_portfolio, minimum_variance_portfolio,
    risk_parity_portfolio,
};

pub use risk::{
    analyze_correlation_risk, calculate_component_es, calculate_component_var,
    monte_carlo_portfolio_var, portfolio_var_parametric, stress_test_portfolio,
};
