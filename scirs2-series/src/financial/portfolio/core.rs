//! Core portfolio structures and basic functionality
//!
//! This module provides fundamental portfolio data structures for managing
//! asset allocations and portfolio metadata. It includes the core Portfolio
//! and PortfolioMetrics structs that form the foundation for portfolio
//! analysis and optimization.

use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;

use crate::error::{Result, TimeSeriesError};

/// Portfolio performance metrics
///
/// Comprehensive performance measurement structure containing key risk and
/// return metrics commonly used in portfolio analysis and reporting.
#[derive(Debug, Clone)]
pub struct PortfolioMetrics<F: Float> {
    /// Total return over the measurement period
    pub total_return: F,
    /// Annualized return (geometric mean)
    pub annualized_return: F,
    /// Annualized volatility (standard deviation of returns)
    pub volatility: F,
    /// Sharpe ratio (risk-adjusted return using total risk)
    pub sharpe_ratio: F,
    /// Sortino ratio (risk-adjusted return using downside risk)
    pub sortino_ratio: F,
    /// Maximum drawdown (worst peak-to-trough decline)
    pub max_drawdown: F,
    /// Calmar ratio (annualized return / maximum drawdown)
    pub calmar_ratio: F,
    /// Value at Risk at 95% confidence level
    pub var_95: F,
    /// Expected Shortfall (Conditional VaR) at 95% confidence level
    pub es_95: F,
}

impl<F: Float> PortfolioMetrics<F> {
    /// Create a new PortfolioMetrics instance
    pub fn new(
        total_return: F,
        annualized_return: F,
        volatility: F,
        sharpe_ratio: F,
        sortino_ratio: F,
        max_drawdown: F,
        calmar_ratio: F,
        var_95: F,
        es_95: F,
    ) -> Self {
        Self {
            total_return,
            annualized_return,
            volatility,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            calmar_ratio,
            var_95,
            es_95,
        }
    }

    /// Get risk-adjusted return ratios
    pub fn risk_adjusted_ratios(&self) -> (F, F, F) {
        (self.sharpe_ratio, self.sortino_ratio, self.calmar_ratio)
    }

    /// Get tail risk measures
    pub fn tail_risk_measures(&self) -> (F, F, F) {
        (self.var_95, self.es_95, self.max_drawdown)
    }

    /// Check if performance metrics are valid
    pub fn is_valid(&self) -> bool {
        self.volatility.is_finite()
            && self.volatility >= F::zero()
            && self.total_return.is_finite()
            && self.annualized_return.is_finite()
    }
}

/// Portfolio weights and holdings
///
/// Core portfolio structure managing asset allocations, names, and
/// rebalancing parameters. Ensures weight constraints are maintained
/// and provides utilities for portfolio manipulation.
#[derive(Debug, Clone)]
pub struct Portfolio<F: Float> {
    /// Asset weights (must sum to approximately 1.0)
    pub weights: Array1<F>,
    /// Asset names/identifiers for tracking
    pub asset_names: Vec<String>,
    /// Optional rebalancing frequency in days
    pub rebalance_frequency: Option<usize>,
}

impl<F: Float + Clone> Portfolio<F> {
    /// Create a new portfolio with given weights and asset names
    ///
    /// # Arguments
    ///
    /// * `weights` - Asset allocation weights (must sum to ~1.0)
    /// * `asset_names` - Corresponding asset identifiers
    ///
    /// # Returns
    ///
    /// * `Result<Self>` - Portfolio instance or error if constraints violated
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scirs2_series::financial::portfolio::core::Portfolio;
    /// use scirs2_core::ndarray::array;
    ///
    /// let weights = array![0.6, 0.4];
    /// let names = vec!["AAPL".to_string(), "GOOGL".to_string()];
    /// let portfolio = Portfolio::new(weights, names).unwrap();
    /// ```
    pub fn new(weights: Array1<F>, asset_names: Vec<String>) -> Result<Self> {
        if weights.len() != asset_names.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: weights.len(),
                actual: asset_names.len(),
            });
        }

        let weight_sum = weights.sum();
        let tolerance = F::from(0.01).unwrap();
        if (weight_sum - F::one()).abs() > tolerance {
            return Err(TimeSeriesError::InvalidInput(
                "Portfolio weights must sum to approximately 1.0".to_string(),
            ));
        }

        Ok(Self {
            weights,
            asset_names,
            rebalance_frequency: None,
        })
    }

    /// Create equally weighted portfolio
    ///
    /// Creates a portfolio where each asset receives equal allocation.
    ///
    /// # Arguments
    ///
    /// * `n_assets` - Number of assets in the portfolio
    /// * `asset_names` - Asset identifiers
    ///
    /// # Returns
    ///
    /// * `Result<Self>` - Equally weighted portfolio
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scirs2_series::financial::portfolio::core::Portfolio;
    ///
    /// let names = vec!["AAPL".to_string(), "GOOGL".to_string(), "MSFT".to_string()];
    /// let portfolio: Portfolio<f64> = Portfolio::equal_weight(3, names).unwrap();
    /// assert_eq!(portfolio.weights[0], 1.0/3.0);
    /// ```
    pub fn equal_weight(n_assets: usize, asset_names: Vec<String>) -> Result<Self> {
        if n_assets == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "Number of assets must be positive".to_string(),
            ));
        }

        if n_assets != asset_names.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n_assets,
                actual: asset_names.len(),
            });
        }

        let weight = F::one() / F::from(n_assets).unwrap();
        let weights = Array1::from_elem(n_assets, weight);

        Self::new(weights, asset_names)
    }

    /// Get portfolio weight for specific asset
    ///
    /// # Arguments
    ///
    /// * `asset_name` - Name of the asset to query
    ///
    /// # Returns
    ///
    /// * `Option<F>` - Weight if asset found, None otherwise
    pub fn get_weight(&self, asset_name: &str) -> Option<F> {
        self.asset_names
            .iter()
            .position(|name| name == asset_name)
            .map(|idx| self.weights[idx])
    }

    /// Set portfolio weight for specific asset
    ///
    /// Updates the weight for a specific asset. Does not automatically
    /// rebalance other weights to maintain sum constraint.
    ///
    /// # Arguments
    ///
    /// * `asset_name` - Name of the asset to update
    /// * `new_weight` - New weight value
    ///
    /// # Returns
    ///
    /// * `Result<()>` - Success or error if asset not found
    pub fn set_weight(&mut self, asset_name: &str, new_weight: F) -> Result<()> {
        if let Some(idx) = self.asset_names.iter().position(|name| name == asset_name) {
            self.weights[idx] = new_weight;
            Ok(())
        } else {
            Err(TimeSeriesError::InvalidInput(format!(
                "Asset '{}' not found in portfolio",
                asset_name
            )))
        }
    }

    /// Get number of assets in portfolio
    pub fn num_assets(&self) -> usize {
        self.asset_names.len()
    }

    /// Validate portfolio constraints
    ///
    /// Checks that weights are non-negative and sum to approximately 1.0
    ///
    /// # Returns
    ///
    /// * `Result<()>` - Success if valid, error otherwise
    pub fn validate(&self) -> Result<()> {
        // Check non-negative weights
        for &weight in self.weights.iter() {
            if weight < F::zero() {
                return Err(TimeSeriesError::InvalidInput(
                    "Portfolio weights must be non-negative".to_string(),
                ));
            }
        }

        // Check sum constraint
        let weight_sum = self.weights.sum();
        let tolerance = F::from(0.01).unwrap();
        if (weight_sum - F::one()).abs() > tolerance {
            return Err(TimeSeriesError::InvalidInput(
                "Portfolio weights must sum to approximately 1.0".to_string(),
            ));
        }

        Ok(())
    }

    /// Normalize weights to sum to 1.0
    ///
    /// Rescales all weights proportionally to ensure they sum to exactly 1.0
    pub fn normalize_weights(&mut self) {
        let weight_sum = self.weights.sum();
        if weight_sum > F::zero() {
            self.weights.mapv_inplace(|w| w / weight_sum);
        }
    }

    /// Set rebalancing frequency
    ///
    /// # Arguments
    ///
    /// * `frequency_days` - Rebalancing frequency in days
    pub fn set_rebalance_frequency(&mut self, frequency_days: usize) {
        self.rebalance_frequency = Some(frequency_days);
    }

    /// Get asset names as slice
    pub fn asset_names(&self) -> &[String] {
        &self.asset_names
    }

    /// Get weights as array view
    pub fn weights(&self) -> &Array1<F> {
        &self.weights
    }
}

/// Calculate portfolio returns from asset returns and weights
///
/// Computes the time series of portfolio returns by taking the weighted
/// average of individual asset returns at each time period.
///
/// # Arguments
///
/// * `asset_returns` - Matrix of asset returns (rows: time, cols: assets)
/// * `weights` - Portfolio weights for each asset
///
/// # Returns
///
/// * `Result<Array1<F>>` - Time series of portfolio returns
///
/// # Examples
///
/// ```rust
/// use scirs2_series::financial::portfolio::core::calculate_portfolio_returns;
/// use scirs2_core::ndarray::{array, Array2};
///
/// let returns = Array2::from_shape_vec((3, 2), vec![0.01, 0.02, -0.01, 0.01, 0.015, -0.005]).unwrap();
/// let weights = array![0.6, 0.4];
/// let portfolio_returns = calculate_portfolio_returns(&returns, &weights).unwrap();
/// ```
pub fn calculate_portfolio_returns<F: Float + Clone>(
    asset_returns: &scirs2_core::ndarray::Array2<F>, // rows: time, cols: assets
    weights: &Array1<F>,
) -> Result<Array1<F>> {
    if asset_returns.ncols() != weights.len() {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: asset_returns.ncols(),
            actual: weights.len(),
        });
    }

    let mut portfolio_returns = Array1::zeros(asset_returns.nrows());

    for t in 0..asset_returns.nrows() {
        let mut return_sum = F::zero();
        for i in 0..weights.len() {
            return_sum = return_sum + weights[i] * asset_returns[[t, i]];
        }
        portfolio_returns[t] = return_sum;
    }

    Ok(portfolio_returns)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{arr1, Array2};

    #[test]
    fn test_portfolio_creation() {
        let weights = arr1(&[0.6, 0.4]);
        let names = vec!["AAPL".to_string(), "GOOGL".to_string()];
        let portfolio = Portfolio::new(weights, names).unwrap();

        assert_eq!(portfolio.num_assets(), 2);
        assert_eq!(portfolio.get_weight("AAPL"), Some(0.6));
        assert_eq!(portfolio.get_weight("GOOGL"), Some(0.4));
    }

    #[test]
    fn test_equal_weight_portfolio() {
        let names = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let portfolio: Portfolio<f64> = Portfolio::equal_weight(3, names).unwrap();

        assert_eq!(portfolio.num_assets(), 3);
        for weight in portfolio.weights.iter() {
            assert!((*weight - 1.0 / 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_portfolio_validation() {
        let weights = arr1(&[0.6, 0.4]);
        let names = vec!["A".to_string(), "B".to_string()];
        let portfolio = Portfolio::new(weights, names).unwrap();

        assert!(portfolio.validate().is_ok());
    }

    #[test]
    fn test_invalid_weights_sum() {
        let weights = arr1(&[0.6, 0.6]); // Sum > 1
        let names = vec!["A".to_string(), "B".to_string()];
        let result = Portfolio::new(weights, names);

        assert!(result.is_err());
    }

    #[test]
    fn test_dimension_mismatch() {
        let weights = arr1(&[0.6, 0.4]);
        let names = vec!["A".to_string()]; // Only one name for two weights
        let result = Portfolio::new(weights, names);

        assert!(result.is_err());
    }

    #[test]
    fn test_calculate_portfolio_returns() {
        let asset_returns =
            Array2::from_shape_vec((3, 2), vec![0.01, 0.02, -0.01, 0.01, 0.015, -0.005]).unwrap();
        let weights = arr1(&[0.6, 0.4]);

        let portfolio_returns = calculate_portfolio_returns(&asset_returns, &weights).unwrap();

        assert_eq!(portfolio_returns.len(), 3);

        // First period: 0.6 * 0.01 + 0.4 * 0.02 = 0.014
        assert!((portfolio_returns[0] - 0.014).abs() < 1e-10);
    }

    #[test]
    fn test_portfolio_metrics_creation() {
        let metrics = PortfolioMetrics::new(0.15, 0.12, 0.18, 0.67, 0.89, 0.08, 1.5, 0.05, 0.07);

        assert!(metrics.is_valid());

        let (sharpe, sortino, calmar) = metrics.risk_adjusted_ratios();
        assert_eq!(sharpe, 0.67);
        assert_eq!(sortino, 0.89);
        assert_eq!(calmar, 1.5);
    }
}
