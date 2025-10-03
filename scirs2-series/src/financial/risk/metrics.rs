//! Core risk metrics for financial analysis
//!
//! This module provides fundamental risk measurement techniques including
//! Value at Risk (VaR), Conditional VaR, and various statistical risk metrics
//! commonly used in quantitative finance and risk management.
//!
//! # Overview
//!
//! Risk metrics are essential for:
//! - Regulatory compliance (Basel III, Solvency II)
//! - Portfolio risk management
//! - Performance evaluation
//! - Capital allocation decisions
//!
//! # Categories
//!
//! ## Value at Risk (VaR)
//! - **Historical Simulation**: Uses historical returns distribution
//! - **Parametric VaR**: Assumes normal distribution of returns
//! - **Monte Carlo VaR**: Uses Monte Carlo simulation
//! - **Rolling VaR**: Time-varying VaR estimation
//!
//! ## Expected Shortfall (ES)
//! Also known as Conditional VaR (CVaR), measures expected loss
//! beyond the VaR threshold.
//!
//! ## Risk-Adjusted Returns
//! - **Sharpe Ratio**: Risk-adjusted return using total risk
//! - **Sortino Ratio**: Risk-adjusted return using downside risk
//! - **Information Ratio**: Risk-adjusted active return
//!
//! ## Market Risk
//! - **Beta**: Systematic risk measure relative to market
//! - **Treynor Ratio**: Risk-adjusted return using systematic risk
//! - **Jensen's Alpha**: Risk-adjusted excess return
//!
//! # Examples
//!
//! ## Basic VaR Calculation
//! ```rust
//! use scirs2_series::financial::risk::metrics::{var_historical, expected_shortfall};
//! use scirs2_core::ndarray::array;
//!
//! let returns = array![-0.05, 0.02, -0.03, 0.01, -0.02, 0.03, -0.01, 0.04];
//! let confidence = 0.95; // 95% confidence level
//!
//! // Historical VaR
//! let var_95 = var_historical(&returns, confidence).unwrap();
//! println!("95% VaR: {:.4}", var_95);
//!
//! // Expected Shortfall (Conditional VaR)
//! let es_95 = expected_shortfall(&returns, confidence).unwrap();
//! println!("95% ES: {:.4}", es_95);
//! ```
//!
//! ## Risk-Adjusted Performance
//! ```rust
//! use scirs2_series::financial::risk::metrics::{sharpe_ratio, sortino_ratio};
//! use scirs2_core::ndarray::array;
//!
//! let returns = array![0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.003];
//! let risk_free_rate = 0.02; // 2% annual
//! let periods_per_year = 252; // Daily data
//!
//! // Sharpe ratio
//! let sharpe = sharpe_ratio(&returns, risk_free_rate, periods_per_year).unwrap();
//!
//! // Sortino ratio (uses downside deviation)
//! let sortino = sortino_ratio(&returns, risk_free_rate, periods_per_year).unwrap();
//! ```
//!
//! ## Market Beta Analysis
//! ```rust
//! use scirs2_series::financial::risk::metrics::{beta, jensens_alpha};
//! use scirs2_core::ndarray::array;
//!
//! let asset_returns = array![0.01, -0.02, 0.015, -0.01, 0.005];
//! let market_returns = array![0.008, -0.015, 0.012, -0.008, 0.004];
//! let risk_free_rate = 0.02;
//! let periods_per_year = 252;
//!
//! // Calculate beta (systematic risk)
//! let beta_coef = beta(&asset_returns, &market_returns).unwrap();
//!
//! // Calculate Jensen's alpha (risk-adjusted excess return)
//! let alpha = jensens_alpha(&asset_returns, &market_returns, risk_free_rate, periods_per_year).unwrap();
//! ```

use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;

use crate::error::{Result, TimeSeriesError};

/// Calculate Value at Risk (VaR) using historical simulation
///
/// Historical VaR uses the empirical distribution of historical returns
/// to estimate potential losses. This non-parametric approach makes no
/// assumptions about the return distribution.
///
/// # Arguments
///
/// * `returns` - Historical return series
/// * `confidence` - Confidence level (e.g., 0.95 for 95% VaR)
///
/// # Returns
///
/// * `Result<F>` - VaR as a positive number (loss magnitude)
///
/// # Examples
///
/// ```rust
/// use scirs2_series::financial::risk::metrics::var_historical;
/// use scirs2_core::ndarray::array;
///
/// let returns = array![-0.05, 0.02, -0.03, 0.01, -0.02, 0.03, -0.01, 0.04];
/// let var_95 = var_historical(&returns, 0.95).unwrap();
/// ```
pub fn var_historical<F: Float + Clone>(returns: &Array1<F>, confidence: f64) -> Result<F> {
    if returns.is_empty() {
        return Err(TimeSeriesError::InvalidInput(
            "Returns cannot be empty".to_string(),
        ));
    }

    if confidence <= 0.0 || confidence >= 1.0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "confidence".to_string(),
            message: "Confidence must be between 0 and 1".to_string(),
        });
    }

    let mut sorted_returns = returns.to_vec();
    sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let index = ((1.0 - confidence) * sorted_returns.len() as f64) as usize;
    let index = index.min(sorted_returns.len() - 1);

    // Return as positive number (loss magnitude)
    Ok(-sorted_returns[index])
}

/// Calculate Expected Shortfall (Conditional VaR)
///
/// Expected Shortfall measures the average loss beyond the VaR threshold.
/// It provides information about tail risk and is a coherent risk measure.
///
/// # Arguments
///
/// * `returns` - Historical return series
/// * `confidence` - Confidence level (e.g., 0.95 for 95% ES)
///
/// # Returns
///
/// * `Result<F>` - Expected Shortfall as a positive number
///
/// # Examples
///
/// ```rust
/// use scirs2_series::financial::risk::metrics::expected_shortfall;
/// use scirs2_core::ndarray::array;
///
/// let returns = array![-0.05, 0.02, -0.03, 0.01, -0.02, 0.03, -0.01, 0.04];
/// let es_95 = expected_shortfall(&returns, 0.95).unwrap();
/// ```
pub fn expected_shortfall<F: Float + Clone + std::iter::Sum>(
    returns: &Array1<F>,
    confidence: f64,
) -> Result<F> {
    let var = var_historical(returns, confidence)?;

    let tail_returns: Vec<F> = returns.iter().filter(|&&x| x <= -var).cloned().collect();

    if tail_returns.is_empty() {
        return Ok(var);
    }

    let sum = tail_returns.iter().fold(F::zero(), |acc, &x| acc + x);
    Ok(-(sum / F::from(tail_returns.len()).unwrap()))
}

/// Calculate Value at Risk using parametric method
///
/// Parametric VaR assumes returns follow a normal distribution and uses
/// the sample mean and standard deviation to estimate VaR.
///
/// # Arguments
///
/// * `returns` - Historical return series
/// * `confidence_level` - Confidence level (e.g., 0.95 for 95% VaR)
///
/// # Returns
///
/// * `Result<F>` - Parametric VaR as a positive number
///
/// # Examples
///
/// ```rust
/// use scirs2_series::financial::risk::metrics::parametric_var;
/// use scirs2_core::ndarray::array;
///
/// let returns = array![0.01, -0.02, 0.015, -0.01, 0.005];
/// let var = parametric_var(&returns, 0.95).unwrap();
/// ```
pub fn parametric_var<F: Float + Clone>(returns: &Array1<F>, confidence_level: F) -> Result<F> {
    if returns.is_empty() {
        return Err(TimeSeriesError::InvalidInput(
            "Returns array cannot be empty".to_string(),
        ));
    }

    let n = returns.len();
    let mean = returns.sum() / F::from(n).unwrap();
    let variance = returns
        .iter()
        .map(|&r| (r - mean).powi(2))
        .fold(F::zero(), |acc, x| acc + x)
        / F::from(n - 1).unwrap();
    let std_dev = variance.sqrt();

    // Normal distribution inverse CDF approximation
    let alpha = F::one() - confidence_level;
    let z_score = normal_inverse_cdf(alpha);
    let var = -(mean + z_score * std_dev);

    Ok(var)
}

/// Calculate Monte Carlo Value at Risk
///
/// Monte Carlo VaR uses simulation to generate potential future returns
/// and estimates VaR from the simulated distribution.
///
/// # Arguments
///
/// * `returns` - Historical return series for parameter estimation
/// * `confidence_level` - Confidence level
/// * `num_simulations` - Number of Monte Carlo simulations
///
/// # Returns
///
/// * `Result<F>` - Monte Carlo VaR as a positive number
pub fn monte_carlo_var<F: Float + Clone>(
    returns: &Array1<F>,
    confidence_level: F,
    num_simulations: usize,
) -> Result<F> {
    if returns.is_empty() {
        return Err(TimeSeriesError::InvalidInput(
            "Returns array cannot be empty".to_string(),
        ));
    }

    let n = returns.len();
    let mean = returns.sum() / F::from(n).unwrap();
    let variance = returns
        .iter()
        .map(|&r| (r - mean).powi(2))
        .fold(F::zero(), |acc, x| acc + x)
        / F::from(n - 1).unwrap();
    let std_dev = variance.sqrt();

    // Simple Monte Carlo simulation using normal distribution
    let mut simulated_returns = Vec::with_capacity(num_simulations);

    // Simple random number generation (in practice, use proper RNG)
    let mut seed = 12345u64;
    for _ in 0..num_simulations {
        // Simple LCG for demonstration
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let u1 = (seed as f64) / (u64::MAX as f64);

        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let u2 = (seed as f64) / (u64::MAX as f64);

        // Box-Muller transformation
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let simulated_return = mean + std_dev * F::from(z).unwrap();
        simulated_returns.push(simulated_return);
    }

    // Sort and find VaR
    simulated_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let var_index = ((F::one() - confidence_level) * F::from(num_simulations).unwrap())
        .to_usize()
        .unwrap();
    let var = if var_index < simulated_returns.len() {
        -simulated_returns[var_index]
    } else {
        -simulated_returns[0]
    };

    Ok(var)
}

/// Calculate Sharpe ratio (excess return / volatility)
///
/// The Sharpe ratio measures risk-adjusted return by dividing excess return
/// by total volatility. Higher values indicate better risk-adjusted performance.
///
/// # Arguments
///
/// * `returns` - Return series
/// * `risk_free_rate` - Annual risk-free rate
/// * `periods_per_year` - Number of periods per year (252 for daily, 12 for monthly)
///
/// # Returns
///
/// * `Result<F>` - Annualized Sharpe ratio
///
/// # Examples
///
/// ```rust
/// use scirs2_series::financial::risk::metrics::sharpe_ratio;
/// use scirs2_core::ndarray::array;
///
/// let returns = array![0.01, -0.02, 0.015, -0.008, 0.012];
/// let sharpe = sharpe_ratio(&returns, 0.02, 252).unwrap();
/// ```
pub fn sharpe_ratio<F: Float + Clone>(
    returns: &Array1<F>,
    risk_free_rate: F,
    periods_per_year: usize,
) -> Result<F> {
    if returns.is_empty() {
        return Err(TimeSeriesError::InvalidInput(
            "Returns cannot be empty".to_string(),
        ));
    }

    // Calculate excess returns
    let annualized_rf = risk_free_rate / F::from(periods_per_year).unwrap();
    let excess_returns: Array1<F> = returns.mapv(|r| r - annualized_rf);

    // Calculate mean excess return
    let mean_excess = excess_returns.sum() / F::from(returns.len()).unwrap();

    // Calculate standard deviation of excess returns
    let variance = excess_returns.mapv(|r| (r - mean_excess).powi(2)).sum()
        / F::from(returns.len() - 1).unwrap();

    let std_dev = variance.sqrt();

    if std_dev == F::zero() {
        Ok(F::infinity())
    } else {
        // Annualize the ratio
        let annualized_excess = mean_excess * F::from(periods_per_year).unwrap();
        let annualized_std = std_dev * F::from(periods_per_year).unwrap().sqrt();
        Ok(annualized_excess / annualized_std)
    }
}

/// Calculate Sortino ratio (excess return / downside deviation)
///
/// The Sortino ratio modifies the Sharpe ratio by using downside deviation
/// instead of total volatility, focusing only on harmful volatility.
///
/// # Arguments
///
/// * `returns` - Return series
/// * `risk_free_rate` - Annual risk-free rate
/// * `periods_per_year` - Number of periods per year
///
/// # Returns
///
/// * `Result<F>` - Annualized Sortino ratio
pub fn sortino_ratio<F: Float + Clone>(
    returns: &Array1<F>,
    risk_free_rate: F,
    periods_per_year: usize,
) -> Result<F> {
    if returns.is_empty() {
        return Err(TimeSeriesError::InvalidInput(
            "Returns cannot be empty".to_string(),
        ));
    }

    // Calculate excess returns
    let annualized_rf = risk_free_rate / F::from(periods_per_year).unwrap();
    let excess_returns: Array1<F> = returns.mapv(|r| r - annualized_rf);

    // Calculate mean excess return
    let mean_excess = excess_returns.sum() / F::from(returns.len()).unwrap();

    // Calculate downside deviation (only negative excess returns)
    let downside_returns: Vec<F> = excess_returns
        .iter()
        .filter(|&&r| r < F::zero())
        .cloned()
        .collect();

    if downside_returns.is_empty() {
        return Ok(F::infinity());
    }

    let downside_variance = downside_returns
        .iter()
        .map(|&r| r.powi(2))
        .fold(F::zero(), |acc, x| acc + x)
        / F::from(downside_returns.len()).unwrap();

    let downside_deviation = downside_variance.sqrt();

    if downside_deviation == F::zero() {
        Ok(F::infinity())
    } else {
        // Annualize the ratio
        let annualized_excess = mean_excess * F::from(periods_per_year).unwrap();
        let annualized_downside = downside_deviation * F::from(periods_per_year).unwrap().sqrt();
        Ok(annualized_excess / annualized_downside)
    }
}

/// Calculate Information ratio (active return / tracking error)
///
/// The Information ratio measures the risk-adjusted active return relative
/// to a benchmark. It shows how much excess return is generated per unit
/// of tracking error.
///
/// # Arguments
///
/// * `portfolio_returns` - Portfolio return series
/// * `benchmark_returns` - Benchmark return series
///
/// # Returns
///
/// * `Result<F>` - Information ratio
pub fn information_ratio<F: Float + Clone>(
    portfolio_returns: &Array1<F>,
    benchmark_returns: &Array1<F>,
) -> Result<F> {
    if portfolio_returns.len() != benchmark_returns.len() {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: portfolio_returns.len(),
            actual: benchmark_returns.len(),
        });
    }

    if portfolio_returns.is_empty() {
        return Err(TimeSeriesError::InvalidInput(
            "Returns cannot be empty".to_string(),
        ));
    }

    // Calculate active returns (portfolio - benchmark)
    let active_returns: Array1<F> = portfolio_returns
        .iter()
        .zip(benchmark_returns.iter())
        .map(|(&p, &b)| p - b)
        .collect();

    // Calculate mean active return
    let mean_active = active_returns.sum() / F::from(active_returns.len()).unwrap();

    // Calculate tracking error (standard deviation of active returns)
    let variance = active_returns.mapv(|r| (r - mean_active).powi(2)).sum()
        / F::from(active_returns.len() - 1).unwrap();

    let tracking_error = variance.sqrt();

    if tracking_error == F::zero() {
        Ok(F::infinity())
    } else {
        Ok(mean_active / tracking_error)
    }
}

/// Calculate Beta coefficient (systematic risk measure)
///
/// Beta measures the systematic risk of an asset relative to the market.
/// Beta > 1 indicates higher volatility than market, Beta < 1 indicates lower.
///
/// # Arguments
///
/// * `asset_returns` - Asset return series
/// * `market_returns` - Market return series
///
/// # Returns
///
/// * `Result<F>` - Beta coefficient
pub fn beta<F: Float + Clone>(asset_returns: &Array1<F>, market_returns: &Array1<F>) -> Result<F> {
    if asset_returns.len() != market_returns.len() {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: asset_returns.len(),
            actual: market_returns.len(),
        });
    }

    if asset_returns.len() < 2 {
        return Err(TimeSeriesError::InsufficientData {
            message: "Need at least 2 observations for beta calculation".to_string(),
            required: 2,
            actual: asset_returns.len(),
        });
    }

    // Calculate means
    let asset_mean = asset_returns.sum() / F::from(asset_returns.len()).unwrap();
    let market_mean = market_returns.sum() / F::from(market_returns.len()).unwrap();

    // Calculate covariance and market variance
    let mut covariance = F::zero();
    let mut market_variance = F::zero();

    for i in 0..asset_returns.len() {
        let asset_dev = asset_returns[i] - asset_mean;
        let market_dev = market_returns[i] - market_mean;

        covariance = covariance + asset_dev * market_dev;
        market_variance = market_variance + market_dev.powi(2);
    }

    let n = F::from(asset_returns.len() - 1).unwrap();
    covariance = covariance / n;
    market_variance = market_variance / n;

    if market_variance == F::zero() {
        Err(TimeSeriesError::InvalidInput(
            "Market returns have zero variance".to_string(),
        ))
    } else {
        Ok(covariance / market_variance)
    }
}

/// Calculate Treynor ratio (excess return / beta)
///
/// The Treynor ratio measures risk-adjusted return using systematic risk (beta)
/// instead of total risk. It shows excess return per unit of systematic risk.
///
/// # Arguments
///
/// * `returns` - Asset return series
/// * `market_returns` - Market return series
/// * `risk_free_rate` - Annual risk-free rate
/// * `periods_per_year` - Number of periods per year
///
/// # Returns
///
/// * `Result<F>` - Treynor ratio
pub fn treynor_ratio<F: Float + Clone>(
    returns: &Array1<F>,
    market_returns: &Array1<F>,
    risk_free_rate: F,
    periods_per_year: usize,
) -> Result<F> {
    // Calculate portfolio beta
    let portfolio_beta = beta(returns, market_returns)?;

    if portfolio_beta == F::zero() {
        return Ok(F::infinity());
    }

    // Calculate annualized excess return
    let annualized_rf = risk_free_rate / F::from(periods_per_year).unwrap();
    let mean_return = returns.sum() / F::from(returns.len()).unwrap();
    let excess_return = mean_return - annualized_rf;
    let annualized_excess = excess_return * F::from(periods_per_year).unwrap();

    Ok(annualized_excess / portfolio_beta)
}

/// Calculate Jensen's alpha (risk-adjusted excess return)
///
/// Jensen's alpha measures the excess return of a portfolio over what would
/// be expected given its beta and the market return (CAPM prediction).
///
/// # Arguments
///
/// * `returns` - Portfolio return series
/// * `market_returns` - Market return series
/// * `risk_free_rate` - Annual risk-free rate
/// * `periods_per_year` - Number of periods per year
///
/// # Returns
///
/// * `Result<F>` - Jensen's alpha (annualized)
pub fn jensens_alpha<F: Float + Clone>(
    returns: &Array1<F>,
    market_returns: &Array1<F>,
    risk_free_rate: F,
    periods_per_year: usize,
) -> Result<F> {
    // Calculate portfolio beta
    let portfolio_beta = beta(returns, market_returns)?;

    // Calculate mean returns
    let annualized_rf = risk_free_rate / F::from(periods_per_year).unwrap();
    let mean_portfolio = returns.sum() / F::from(returns.len()).unwrap();
    let mean_market = market_returns.sum() / F::from(market_returns.len()).unwrap();

    // Calculate alpha using CAPM formula
    // Alpha = Portfolio Return - (Risk Free Rate + Beta * (Market Return - Risk Free Rate))
    let portfolio_excess = mean_portfolio - annualized_rf;
    let market_excess = mean_market - annualized_rf;
    let expected_excess = portfolio_beta * market_excess;

    Ok((portfolio_excess - expected_excess) * F::from(periods_per_year).unwrap())
}

/// Calculate Omega ratio (probability-weighted gains over losses)
///
/// The Omega ratio considers the entire return distribution and measures
/// the probability-weighted gains above a threshold relative to
/// probability-weighted losses below the threshold.
///
/// # Arguments
///
/// * `returns` - Return series
/// * `threshold` - Threshold return (often risk-free rate)
///
/// # Returns
///
/// * `Result<F>` - Omega ratio
pub fn omega_ratio<F: Float + Clone>(returns: &Array1<F>, threshold: F) -> Result<F> {
    if returns.is_empty() {
        return Err(TimeSeriesError::InvalidInput(
            "Returns cannot be empty".to_string(),
        ));
    }

    let mut gains = F::zero();
    let mut losses = F::zero();

    for &ret in returns.iter() {
        let excess = ret - threshold;
        if excess > F::zero() {
            gains = gains + excess;
        } else {
            losses = losses - excess; // Make positive
        }
    }

    if losses == F::zero() {
        Ok(F::infinity())
    } else {
        Ok(gains / losses)
    }
}

/// Approximate normal inverse CDF using Beasley-Springer-Moro algorithm
fn normal_inverse_cdf<F: Float>(p: F) -> F {
    let a0 = F::from(2.515517).unwrap();
    let a1 = F::from(0.802853).unwrap();
    let a2 = F::from(0.010328).unwrap();
    let b1 = F::from(1.432788).unwrap();
    let b2 = F::from(0.189269).unwrap();
    let b3 = F::from(0.001308).unwrap();

    let t = if p < F::from(0.5).unwrap() {
        (-F::from(2.0).unwrap() * (p.ln())).sqrt()
    } else {
        (-F::from(2.0).unwrap() * ((F::one() - p).ln())).sqrt()
    };

    let numerator = a0 + a1 * t + a2 * t.powi(2);
    let denominator = F::one() + b1 * t + b2 * t.powi(2) + b3 * t.powi(3);
    let z = t - numerator / denominator;

    if p < F::from(0.5).unwrap() {
        -z
    } else {
        z
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr1;

    #[test]
    fn test_var_historical() {
        let returns = arr1(&[-0.05, 0.02, -0.03, 0.01, -0.02, 0.03, -0.01, 0.04]);
        let var = var_historical(&returns, 0.95).unwrap();

        // VaR should be positive (loss magnitude)
        assert!(var > 0.0);

        // VaR should be reasonable for this data
        assert!(var < 0.1);
    }

    #[test]
    fn test_expected_shortfall() {
        let returns = arr1(&[-0.05, 0.02, -0.03, 0.01, -0.02, 0.03, -0.01, 0.04]);
        let var = var_historical(&returns, 0.95).unwrap();
        let es = expected_shortfall(&returns, 0.95).unwrap();

        // ES should be >= VaR
        assert!(es >= var);
        assert!(es > 0.0);
    }

    #[test]
    fn test_parametric_var() {
        let returns = arr1(&[0.01, -0.02, 0.015, -0.01, 0.005]);
        let var = parametric_var(&returns, 0.95).unwrap();

        assert!(var > 0.0);
    }

    #[test]
    fn test_sharpe_ratio() {
        let returns = arr1(&[0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.003]);
        let result = sharpe_ratio(&returns, 0.02, 252);
        assert!(result.is_ok());

        let sharpe = result.unwrap();
        // Sharpe ratio should be finite
        assert!(sharpe.is_finite());
    }

    #[test]
    fn test_sortino_ratio() {
        let returns = arr1(&[0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.003]);
        let result = sortino_ratio(&returns, 0.02, 252);
        assert!(result.is_ok());

        let sortino = result.unwrap();
        // Sortino ratio should be finite or infinity (if no downside)
        assert!(sortino.is_finite() || sortino.is_infinite());
    }

    #[test]
    fn test_beta() {
        let asset_returns = arr1(&[0.01, -0.02, 0.015, -0.01, 0.005]);
        let market_returns = arr1(&[0.008, -0.015, 0.012, -0.008, 0.004]);

        let result = beta(&asset_returns, &market_returns);
        assert!(result.is_ok());

        let beta_coef = result.unwrap();
        // Beta should be reasonable (typically between -3 and 3)
        assert!(beta_coef.abs() < 5.0);
    }

    #[test]
    fn test_information_ratio() {
        let portfolio_returns = arr1(&[0.01, -0.02, 0.015, -0.01, 0.005]);
        let benchmark_returns = arr1(&[0.008, -0.018, 0.012, -0.009, 0.003]);

        let result = information_ratio(&portfolio_returns, &benchmark_returns);
        assert!(result.is_ok());

        let ir = result.unwrap();
        assert!(ir.is_finite() || ir.is_infinite());
    }

    #[test]
    fn test_jensens_alpha() {
        let returns = arr1(&[0.01, -0.02, 0.015, -0.01, 0.005]);
        let market_returns = arr1(&[0.008, -0.015, 0.012, -0.008, 0.004]);

        let result = jensens_alpha(&returns, &market_returns, 0.02, 252);
        assert!(result.is_ok());

        let alpha = result.unwrap();
        // Alpha should be finite
        assert!(alpha.is_finite());
    }

    #[test]
    fn test_omega_ratio() {
        let returns = arr1(&[0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.003]);
        let threshold = 0.0;

        let result = omega_ratio(&returns, threshold);
        assert!(result.is_ok());

        let omega = result.unwrap();
        assert!(omega > 0.0);
    }

    #[test]
    fn test_dimension_mismatch() {
        let returns1 = arr1(&[0.01, -0.02]);
        let returns2 = arr1(&[0.008, -0.015, 0.012]);

        let result = beta(&returns1, &returns2);
        assert!(result.is_err());
    }

    #[test]
    fn test_insufficient_data() {
        let returns = arr1(&[0.01]);
        let market_returns = arr1(&[0.008]);

        let result = beta(&returns, &market_returns);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_confidence() {
        let returns = arr1(&[0.01, -0.02, 0.015]);

        // Invalid confidence > 1
        let result = var_historical(&returns, 1.1);
        assert!(result.is_err());

        // Invalid confidence <= 0
        let result = var_historical(&returns, 0.0);
        assert!(result.is_err());
    }
}
