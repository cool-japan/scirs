//! Portfolio risk analysis and measurement
//!
//! This module provides specialized risk analysis tools for portfolios,
//! including Value at Risk calculations, stress testing, and scenario analysis.
//! These complement the general risk metrics in the parent risk module.

use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;

use crate::error::{Result, TimeSeriesError};

/// Calculate portfolio Value at Risk using parametric method
///
/// Estimates potential portfolio losses using the normal distribution assumption.
/// This method is fast but may underestimate tail risks during extreme market conditions.
///
/// # Arguments
///
/// * `portfolio_value` - Current portfolio market value
/// * `portfolio_return_mean` - Mean portfolio return (per period)
/// * `portfolio_return_std` - Standard deviation of portfolio returns
/// * `confidence_level` - Confidence level (0.90, 0.95, 0.99 typical)
/// * `time_horizon` - Number of periods for VaR calculation
///
/// # Returns
///
/// * `Result<F>` - Estimated Value at Risk (loss amount)
///
/// # Examples
///
/// ```rust
/// use scirs2_series::financial::portfolio::risk::portfolio_var_parametric;
///
/// let portfolio_value = 1000000.0; // $1M portfolio
/// let mean_return = 0.0008; // 0.08% daily return
/// let std_return = 0.015; // 1.5% daily volatility
/// let confidence = 0.95; // 95% confidence
/// let horizon = 1; // 1-day VaR
///
/// let var = portfolio_var_parametric(portfolio_value, mean_return, std_return, confidence, horizon).unwrap();
/// println!("1-day 95% VaR: ${:.2}", var);
/// ```
pub fn portfolio_var_parametric<F: Float + Clone>(
    portfolio_value: F,
    portfolio_return_mean: F,
    portfolio_return_std: F,
    confidence_level: f64,
    time_horizon: usize,
) -> Result<F> {
    if confidence_level <= 0.0 || confidence_level >= 1.0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "confidence_level".to_string(),
            message: "Confidence level must be between 0 and 1".to_string(),
        });
    }

    if time_horizon == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "time_horizon".to_string(),
            message: "Time horizon must be positive".to_string(),
        });
    }

    // Standard normal quantiles for common confidence levels
    let z_score = match confidence_level {
        c if c >= 0.99 => F::from(-2.326).unwrap(), // 99% VaR
        c if c >= 0.975 => F::from(-1.96).unwrap(), // 97.5% VaR
        c if c >= 0.95 => F::from(-1.645).unwrap(), // 95% VaR
        c if c >= 0.90 => F::from(-1.282).unwrap(), // 90% VaR
        _ => {
            // For other confidence levels, use approximation
            let p = 1.0 - confidence_level;
            F::from(normal_inverse_cdf(p)).unwrap()
        }
    };

    // Scale for time horizon (square root rule)
    let horizon_scaling = F::from(time_horizon).unwrap().sqrt();
    let horizon_mean = portfolio_return_mean * F::from(time_horizon).unwrap();
    let horizon_std = portfolio_return_std * horizon_scaling;

    // Calculate VaR
    let var_return = horizon_mean + z_score * horizon_std;
    let var_amount = portfolio_value * var_return.abs();

    Ok(var_amount)
}

/// Calculate portfolio Component Value at Risk (Component VaR)
///
/// Decomposes portfolio VaR into contributions from individual assets,
/// helping identify the primary sources of portfolio risk.
///
/// # Arguments
///
/// * `weights` - Portfolio asset weights
/// * `asset_returns` - Historical returns matrix (rows: time, cols: assets)
/// * `confidence_level` - Confidence level for VaR calculation
///
/// # Returns
///
/// * `Result<Array1<F>>` - Component VaR for each asset
pub fn calculate_component_var<F: Float + Clone + std::iter::Sum>(
    weights: &Array1<F>,
    asset_returns: &scirs2_core::ndarray::Array2<F>,
    confidence_level: f64,
) -> Result<Array1<F>> {
    if asset_returns.ncols() != weights.len() {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: asset_returns.ncols(),
            actual: weights.len(),
        });
    }

    // Calculate portfolio returns
    let portfolio_returns = super::core::calculate_portfolio_returns(asset_returns, weights)?;

    // Calculate portfolio VaR
    let portfolio_var =
        crate::financial::risk::var_historical(&portfolio_returns, confidence_level)?;

    let mut component_vars = Array1::zeros(weights.len());

    // Calculate marginal VaR for each asset
    for i in 0..weights.len() {
        // Small perturbation for numerical derivative
        let epsilon = F::from(0.001).unwrap();
        let mut perturbed_weights = weights.clone();
        perturbed_weights[i] = perturbed_weights[i] + epsilon;

        // Renormalize weights
        let weight_sum = perturbed_weights.sum();
        perturbed_weights.mapv_inplace(|w| w / weight_sum);

        // Calculate perturbed portfolio returns
        let perturbed_returns =
            super::core::calculate_portfolio_returns(asset_returns, &perturbed_weights)?;

        // Calculate perturbed VaR
        let perturbed_var =
            crate::financial::risk::var_historical(&perturbed_returns, confidence_level)?;

        // Marginal VaR (derivative)
        let marginal_var = (perturbed_var - portfolio_var) / epsilon;

        // Component VaR = weight × marginal VaR
        component_vars[i] = weights[i] * marginal_var;
    }

    Ok(component_vars)
}

/// Calculate portfolio stress testing scenarios
///
/// Applies various stress scenarios to evaluate portfolio performance
/// under extreme market conditions.
///
/// # Arguments
///
/// * `weights` - Portfolio asset weights
/// * `asset_returns` - Historical returns matrix
/// * `stress_factors` - Stress multipliers for each scenario
///
/// # Returns
///
/// * `Result<Array1<F>>` - Portfolio returns under stress scenarios
pub fn stress_test_portfolio<F: Float + Clone>(
    weights: &Array1<F>,
    asset_returns: &scirs2_core::ndarray::Array2<F>,
    stress_factors: &Array1<F>,
) -> Result<Array1<F>> {
    if asset_returns.ncols() != weights.len() {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: asset_returns.ncols(),
            actual: weights.len(),
        });
    }

    let mut stressed_returns = Array1::zeros(stress_factors.len());

    for scenario in 0..stress_factors.len() {
        let stress_factor = stress_factors[scenario];
        let mut portfolio_return = F::zero();

        // Apply stress factor to all assets and calculate portfolio return
        for asset in 0..weights.len() {
            // Use worst historical return for this asset and apply stress
            let asset_col = asset_returns.column(asset);
            let min_return =
                asset_col
                    .iter()
                    .fold(F::infinity(), |min, &val| if val < min { val } else { min });

            let stressed_asset_return = min_return * stress_factor;
            portfolio_return = portfolio_return + weights[asset] * stressed_asset_return;
        }

        stressed_returns[scenario] = portfolio_return;
    }

    Ok(stressed_returns)
}

/// Calculate Expected Shortfall decomposition
///
/// Decomposes portfolio Expected Shortfall into asset contributions,
/// providing insight into tail risk sources.
///
/// # Arguments
///
/// * `weights` - Portfolio asset weights
/// * `asset_returns` - Historical returns matrix
/// * `confidence_level` - Confidence level for ES calculation
///
/// # Returns
///
/// * `Result<Array1<F>>` - Component Expected Shortfall for each asset
pub fn calculate_component_es<F: Float + Clone + std::iter::Sum>(
    weights: &Array1<F>,
    asset_returns: &scirs2_core::ndarray::Array2<F>,
    confidence_level: f64,
) -> Result<Array1<F>> {
    if asset_returns.ncols() != weights.len() {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: asset_returns.ncols(),
            actual: weights.len(),
        });
    }

    // Calculate portfolio returns
    let portfolio_returns = super::core::calculate_portfolio_returns(asset_returns, weights)?;

    // Calculate portfolio ES
    let portfolio_es =
        crate::financial::risk::expected_shortfall(&portfolio_returns, confidence_level)?;

    let mut component_es = Array1::zeros(weights.len());

    // Calculate marginal ES for each asset
    for i in 0..weights.len() {
        // Small perturbation for numerical derivative
        let epsilon = F::from(0.001).unwrap();
        let mut perturbed_weights = weights.clone();
        perturbed_weights[i] = perturbed_weights[i] + epsilon;

        // Renormalize weights
        let weight_sum = perturbed_weights.sum();
        perturbed_weights.mapv_inplace(|w| w / weight_sum);

        // Calculate perturbed portfolio returns
        let perturbed_returns =
            super::core::calculate_portfolio_returns(asset_returns, &perturbed_weights)?;

        // Calculate perturbed ES
        let perturbed_es =
            crate::financial::risk::expected_shortfall(&perturbed_returns, confidence_level)?;

        // Marginal ES (derivative)
        let marginal_es = (perturbed_es - portfolio_es) / epsilon;

        // Component ES = weight × marginal ES
        component_es[i] = weights[i] * marginal_es;
    }

    Ok(component_es)
}

/// Calculate portfolio correlation-based risk measures
///
/// Analyzes how portfolio risk changes based on asset correlations,
/// useful for understanding diversification benefits.
///
/// # Arguments
///
/// * `weights` - Portfolio asset weights
/// * `correlation_matrix` - Asset correlation matrix
/// * `individual_volatilities` - Individual asset volatilities
///
/// # Returns
///
/// * `Result<(F, F, F)>` - (portfolio_volatility, diversification_ratio, concentration_risk)
pub fn analyze_correlation_risk<F: Float + Clone>(
    weights: &Array1<F>,
    correlation_matrix: &scirs2_core::ndarray::Array2<F>,
    individual_volatilities: &Array1<F>,
) -> Result<(F, F, F)> {
    let n = weights.len();

    if correlation_matrix.nrows() != n || correlation_matrix.ncols() != n {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: n,
            actual: correlation_matrix.nrows(),
        });
    }

    if individual_volatilities.len() != n {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: n,
            actual: individual_volatilities.len(),
        });
    }

    // Calculate portfolio volatility
    let mut portfolio_variance = F::zero();
    for i in 0..n {
        for j in 0..n {
            let correlation = correlation_matrix[[i, j]];
            portfolio_variance = portfolio_variance
                + weights[i]
                    * weights[j]
                    * individual_volatilities[i]
                    * individual_volatilities[j]
                    * correlation;
        }
    }
    let portfolio_volatility = portfolio_variance.sqrt();

    // Calculate weighted average of individual volatilities
    let weighted_avg_volatility = weights
        .iter()
        .zip(individual_volatilities.iter())
        .map(|(&w, &vol)| w * vol)
        .fold(F::zero(), |acc, x| acc + x);

    // Diversification ratio
    let diversification_ratio = if portfolio_volatility > F::zero() {
        weighted_avg_volatility / portfolio_volatility
    } else {
        F::one()
    };

    // Concentration risk (Herfindahl index)
    let concentration_risk = weights.mapv(|w| w.powi(2)).sum();

    Ok((
        portfolio_volatility,
        diversification_ratio,
        concentration_risk,
    ))
}

/// Perform Monte Carlo VaR simulation for portfolio
///
/// Uses Monte Carlo simulation to estimate portfolio VaR under
/// various distributional assumptions.
///
/// # Arguments
///
/// * `weights` - Portfolio asset weights
/// * `expected_returns` - Expected returns for each asset
/// * `covariance_matrix` - Asset return covariance matrix
/// * `confidence_level` - Confidence level for VaR
/// * `num_simulations` - Number of Monte Carlo simulations
///
/// # Returns
///
/// * `Result<F>` - Monte Carlo VaR estimate
pub fn monte_carlo_portfolio_var<F: Float + Clone>(
    weights: &Array1<F>,
    expected_returns: &Array1<F>,
    covariance_matrix: &scirs2_core::ndarray::Array2<F>,
    confidence_level: f64,
    num_simulations: usize,
) -> Result<F> {
    let n = weights.len();

    if expected_returns.len() != n {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: n,
            actual: expected_returns.len(),
        });
    }

    if covariance_matrix.nrows() != n || covariance_matrix.ncols() != n {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: n,
            actual: covariance_matrix.nrows(),
        });
    }

    let mut simulated_returns = Vec::with_capacity(num_simulations);

    // Simple simulation using multivariate normal (simplified)
    let mut seed = 42u64;

    for _ in 0..num_simulations {
        let mut portfolio_return = F::zero();

        // Generate correlated random returns (simplified approach)
        for i in 0..n {
            // Generate standard normal
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let u1 = (seed as f64) / (u64::MAX as f64);

            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let u2 = (seed as f64) / (u64::MAX as f64);

            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

            // Scale by volatility and add expected return
            let individual_volatility = covariance_matrix[[i, i]].sqrt();
            let simulated_return =
                expected_returns[i] + individual_volatility * F::from(z).unwrap();

            portfolio_return = portfolio_return + weights[i] * simulated_return;
        }

        simulated_returns.push(portfolio_return);
    }

    // Sort and find VaR
    simulated_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let var_index = ((1.0 - confidence_level) * num_simulations as f64) as usize;
    let var_index = var_index.min(simulated_returns.len() - 1);

    Ok(-simulated_returns[var_index]) // Return as positive loss
}

/// Approximate normal inverse CDF for z-score calculation
fn normal_inverse_cdf(p: f64) -> f64 {
    // Beasley-Springer-Moro approximation
    let a0 = 2.515517;
    let a1 = 0.802853;
    let a2 = 0.010328;
    let b1 = 1.432788;
    let b2 = 0.189269;
    let b3 = 0.001308;

    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };

    let numerator = a0 + a1 * t + a2 * t.powi(2);
    let denominator = 1.0 + b1 * t + b2 * t.powi(2) + b3 * t.powi(3);
    let z = t - numerator / denominator;

    if p < 0.5 {
        -z
    } else {
        z
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{arr1, Array2};

    #[test]
    fn test_portfolio_var_parametric() {
        let portfolio_value = 1000000.0;
        let mean_return = 0.0008;
        let std_return = 0.015;
        let confidence = 0.95;
        let horizon = 1;

        let result = portfolio_var_parametric(
            portfolio_value,
            mean_return,
            std_return,
            confidence,
            horizon,
        );
        assert!(result.is_ok());

        let var = result.unwrap();
        assert!(var > 0.0);
        assert!(var < portfolio_value); // VaR should be less than portfolio value
    }

    #[test]
    fn test_component_var() {
        let weights = arr1(&[0.6, 0.4]);
        let returns = Array2::from_shape_vec(
            (5, 2),
            vec![
                0.01, 0.02, -0.01, 0.005, 0.015, -0.008, 0.005, 0.012, -0.002, 0.008,
            ],
        )
        .unwrap();
        let confidence = 0.95;

        let result = calculate_component_var(&weights, &returns, confidence);
        assert!(result.is_ok());

        let comp_var = result.unwrap();
        assert_eq!(comp_var.len(), 2);

        // Component VaR values should be finite
        let total_comp_var = comp_var.sum();
        assert!(total_comp_var.is_finite());
    }

    #[test]
    fn test_stress_test_portfolio() {
        let weights = arr1(&[0.5, 0.5]);
        let returns = Array2::from_shape_vec(
            (4, 2),
            vec![0.01, 0.02, -0.01, 0.005, 0.015, -0.008, 0.005, 0.012],
        )
        .unwrap();
        let stress_factors = arr1(&[1.5, 2.0, 3.0]); // 50%, 100%, 200% stress

        let result = stress_test_portfolio(&weights, &returns, &stress_factors);
        assert!(result.is_ok());

        let stressed_returns = result.unwrap();
        assert_eq!(stressed_returns.len(), 3);

        // Stressed returns should generally be negative and increasing in magnitude
        for &ret in stressed_returns.iter() {
            assert!(ret <= 0.0);
        }
    }

    #[test]
    fn test_correlation_risk_analysis() {
        let weights = arr1(&[0.5, 0.5]);
        let correlation_matrix = Array2::from_shape_vec((2, 2), vec![1.0, 0.3, 0.3, 1.0]).unwrap();
        let volatilities = arr1(&[0.15, 0.20]);

        let result = analyze_correlation_risk(&weights, &correlation_matrix, &volatilities);
        assert!(result.is_ok());

        let (portfolio_vol, div_ratio, concentration) = result.unwrap();
        assert!(portfolio_vol > 0.0);
        assert!(div_ratio >= 1.0); // Diversification ratio should be >= 1
        assert!(concentration > 0.0 && concentration <= 1.0);
    }

    #[test]
    fn test_monte_carlo_var() {
        let weights = arr1(&[0.6, 0.4]);
        let expected_returns = arr1(&[0.08, 0.12]);
        let cov_matrix = Array2::from_shape_vec((2, 2), vec![0.01, 0.002, 0.002, 0.015]).unwrap();
        let confidence = 0.95;
        let num_simulations = 1000;

        let result = monte_carlo_portfolio_var(
            &weights,
            &expected_returns,
            &cov_matrix,
            confidence,
            num_simulations,
        );
        assert!(result.is_ok());

        let var = result.unwrap();
        assert!(var >= 0.0);
    }

    #[test]
    fn test_invalid_confidence_level() {
        let result = portfolio_var_parametric(100000.0, 0.001, 0.02, 1.1, 1);
        assert!(result.is_err());

        let result = portfolio_var_parametric(100000.0, 0.001, 0.02, 0.0, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_time_horizon() {
        let result = portfolio_var_parametric(100000.0, 0.001, 0.02, 0.95, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_dimension_mismatches() {
        let weights = arr1(&[0.6, 0.4]);
        let returns = Array2::from_shape_vec((3, 3), vec![0.0; 9]).unwrap();

        let result = calculate_component_var(&weights, &returns, 0.95);
        assert!(result.is_err());
    }
}
