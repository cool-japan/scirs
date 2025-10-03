//! Financial modeling and quantitative finance domain metrics
//!
//! This module provides specialized metrics for financial applications including:
//! - Risk management and Value at Risk (VaR) calculations
//! - Portfolio optimization and performance evaluation
//! - Credit risk modeling and default prediction
//! - Market risk assessment and stress testing
//! - Algorithmic trading strategy evaluation
//! - Fraud detection and anti-money laundering
//! - Regulatory compliance and Basel III metrics
//! - ESG (Environmental, Social, Governance) scoring

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use super::{DomainEvaluationResult, DomainMetrics};
use crate::error::{MetricsError, Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::Float;
use std::collections::HashMap;

/// Comprehensive financial metrics suite
#[derive(Debug)]
pub struct FinancialSuite {
    /// Risk management metrics
    pub risk_metrics: RiskManagementMetrics,
    /// Portfolio performance metrics
    pub portfolio_metrics: PortfolioMetrics,
    /// Credit risk metrics
    pub credit_metrics: CreditRiskMetrics,
    /// Market risk metrics
    pub market_risk_metrics: MarketRiskMetrics,
    /// Trading strategy metrics
    pub trading_metrics: TradingStrategyMetrics,
    /// Fraud detection metrics
    pub fraud_metrics: FraudDetectionMetrics,
    /// Regulatory compliance metrics
    pub regulatory_metrics: RegulatoryMetrics,
    /// ESG scoring metrics
    pub esg_metrics: ESGMetrics,
}

impl Default for FinancialSuite {
    fn default() -> Self {
        Self::new()
    }
}

impl FinancialSuite {
    /// Create new financial metrics suite
    pub fn new() -> Self {
        Self {
            risk_metrics: RiskManagementMetrics::new(),
            portfolio_metrics: PortfolioMetrics::new(),
            credit_metrics: CreditRiskMetrics::new(),
            market_risk_metrics: MarketRiskMetrics::new(),
            trading_metrics: TradingStrategyMetrics::new(),
            fraud_metrics: FraudDetectionMetrics::new(),
            regulatory_metrics: RegulatoryMetrics::new(),
            esg_metrics: ESGMetrics::new(),
        }
    }

    /// Get risk management metrics
    pub fn risk_management(&self) -> &RiskManagementMetrics {
        &self.risk_metrics
    }

    /// Get portfolio metrics
    pub fn portfolio(&self) -> &PortfolioMetrics {
        &self.portfolio_metrics
    }

    /// Get credit risk metrics
    pub fn credit_risk(&self) -> &CreditRiskMetrics {
        &self.credit_metrics
    }

    /// Get market risk metrics
    pub fn market_risk(&self) -> &MarketRiskMetrics {
        &self.market_risk_metrics
    }

    /// Get trading strategy metrics
    pub fn trading_strategy(&self) -> &TradingStrategyMetrics {
        &self.trading_metrics
    }

    /// Get fraud detection metrics
    pub fn fraud_detection(&self) -> &FraudDetectionMetrics {
        &self.fraud_metrics
    }

    /// Get regulatory compliance metrics
    pub fn regulatory(&self) -> &RegulatoryMetrics {
        &self.regulatory_metrics
    }

    /// Get ESG metrics
    pub fn esg(&self) -> &ESGMetrics {
        &self.esg_metrics
    }
}

impl DomainMetrics for FinancialSuite {
    type Result = DomainEvaluationResult;

    fn domain_name(&self) -> &'static str {
        "Financial Modeling & Quantitative Finance"
    }

    fn available_metrics(&self) -> Vec<&'static str> {
        vec![
            "value_at_risk",
            "conditional_value_at_risk",
            "expected_shortfall",
            "maximum_drawdown",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "information_ratio",
            "treynor_ratio",
            "jensen_alpha",
            "beta",
            "tracking_error",
            "upside_capture",
            "downside_capture",
            "probability_of_default",
            "loss_given_default",
            "exposure_at_default",
            "expected_credit_loss",
            "credit_var",
            "discriminatory_power",
            "population_stability_index",
            "kolmogorov_smirnov",
            "gini_coefficient",
            "area_under_curve",
            "concordance_ratio",
            "divergence_ratio",
            "total_return",
            "excess_return",
            "annualized_return",
            "volatility",
            "skewness",
            "kurtosis",
            "hit_ratio",
            "profit_factor",
            "recovery_factor",
            "ulcer_index",
            "sterling_ratio",
            "burke_ratio",
            "fraud_detection_rate",
            "false_positive_rate",
            "alert_precision",
            "investigation_efficiency",
            "regulatory_capital_ratio",
            "leverage_ratio",
            "liquidity_coverage_ratio",
            "net_stable_funding_ratio",
            "esg_score",
            "carbon_intensity",
            "social_impact_score",
            "governance_score",
        ]
    }

    fn metric_descriptions(&self) -> HashMap<&'static str, &'static str> {
        let mut descriptions = HashMap::new();
        descriptions.insert(
            "value_at_risk",
            "Maximum potential loss over a given time period at a specific confidence level",
        );
        descriptions.insert(
            "conditional_value_at_risk",
            "Expected loss in the worst-case scenarios beyond VaR",
        );
        descriptions.insert(
            "sharpe_ratio",
            "Risk-adjusted return measure (excess return per unit of volatility)",
        );
        descriptions.insert(
            "sortino_ratio",
            "Modified Sharpe ratio that only considers downside volatility",
        );
        descriptions.insert(
            "maximum_drawdown",
            "Largest peak-to-trough decline in portfolio value",
        );
        descriptions.insert(
            "probability_of_default",
            "Likelihood that a borrower will default within a given time period",
        );
        descriptions.insert(
            "expected_credit_loss",
            "Expected monetary loss from credit defaults",
        );
        descriptions.insert(
            "discriminatory_power",
            "Ability of a model to distinguish between good and bad credit risks",
        );
        descriptions.insert(
            "gini_coefficient",
            "Measure of inequality or discrimination power in credit models",
        );
        descriptions.insert(
            "hit_ratio",
            "Percentage of profitable trades in a trading strategy",
        );
        descriptions.insert("profit_factor", "Ratio of gross profit to gross loss");
        descriptions.insert(
            "fraud_detection_rate",
            "Percentage of fraudulent transactions correctly identified",
        );
        descriptions.insert(
            "regulatory_capital_ratio",
            "Ratio of bank's capital to risk-weighted assets",
        );
        descriptions.insert(
            "esg_score",
            "Environmental, Social, and Governance performance score",
        );
        descriptions
    }
}

/// Risk management and VaR calculation metrics
#[derive(Debug, Clone)]
pub struct RiskManagementMetrics {
    /// Historical VaR calculations
    pub var_results: HashMap<String, f64>,
    /// Stress test results
    pub stress_test_results: HashMap<String, f64>,
    /// Risk decomposition
    pub risk_decomposition: HashMap<String, f64>,
}

impl RiskManagementMetrics {
    pub fn new() -> Self {
        Self {
            var_results: HashMap::new(),
            stress_test_results: HashMap::new(),
            risk_decomposition: HashMap::new(),
        }
    }

    /// Calculate Value at Risk using historical simulation
    pub fn historical_var<F>(
        &self,
        returns: &Array1<F>,
        confidencelevel: F,
        holding_period: usize,
    ) -> Result<F>
    where
        F: Float + scirs2_core::numeric::FromPrimitive + PartialOrd + Clone + std::iter::Sum,
    {
        if returns.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Returns array is empty".to_string(),
            ));
        }

        // Sort returns in ascending order
        let mut sorted_returns: Vec<F> = returns.iter().cloned().collect();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate percentile index
        let alpha = F::one() - confidencelevel;
        let index = (alpha * F::from(sorted_returns.len()).unwrap()).floor();
        let var_index = index.to_usize().unwrap_or(0);

        if var_index < sorted_returns.len() {
            let daily_var = -sorted_returns[var_index]; // VaR is positive for losses
                                                        // Scale for holding _period (assuming sqrt of time scaling)
            let holding_period_factor = F::from(holding_period).unwrap().sqrt();
            Ok(daily_var * holding_period_factor)
        } else {
            Ok(F::zero())
        }
    }

    /// Calculate Conditional Value at Risk (Expected Shortfall)
    pub fn conditional_var<F>(&self, returns: &Array1<F>, confidencelevel: F) -> Result<F>
    where
        F: Float + scirs2_core::numeric::FromPrimitive + PartialOrd + Clone + std::iter::Sum,
    {
        if returns.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Returns array is empty".to_string(),
            ));
        }

        let mut sorted_returns: Vec<F> = returns.iter().cloned().collect();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let alpha = F::one() - confidencelevel;
        let cutoff_index = (alpha * F::from(sorted_returns.len()).unwrap())
            .floor()
            .to_usize()
            .unwrap_or(0);

        if cutoff_index == 0 {
            return Ok(F::zero());
        }

        // Calculate mean of worst returns (tail)
        let tail_sum: F = sorted_returns[..cutoff_index].iter().cloned().sum();
        let cvar = -tail_sum / F::from(cutoff_index).unwrap(); // CVaR is positive for losses

        Ok(cvar)
    }

    /// Calculate Maximum Drawdown
    pub fn maximum_drawdown<F>(&self, prices: &Array1<F>) -> Result<F>
    where
        F: Float + scirs2_core::numeric::FromPrimitive + PartialOrd,
    {
        if prices.len() < 2 {
            return Ok(F::zero());
        }

        let mut peak = prices[0];
        let mut max_dd = F::zero();

        for &price in prices.iter().skip(1) {
            if price > peak {
                peak = price;
            }

            let drawdown = (peak - price) / peak;
            if drawdown > max_dd {
                max_dd = drawdown;
            }
        }

        Ok(max_dd)
    }

    /// Calculate portfolio beta
    pub fn beta<F>(&self, portfolioreturns: &Array1<F>, marketreturns: &Array1<F>) -> Result<F>
    where
        F: Float + scirs2_core::numeric::FromPrimitive + std::iter::Sum,
    {
        if portfolioreturns.len() != marketreturns.len() {
            return Err(MetricsError::InvalidInput(
                "Portfolio and market _returns must have same length".to_string(),
            ));
        }

        let portfolio_mean =
            portfolioreturns.iter().cloned().sum::<F>() / F::from(portfolioreturns.len()).unwrap();
        let market_mean =
            marketreturns.iter().cloned().sum::<F>() / F::from(marketreturns.len()).unwrap();

        let covariance = portfolioreturns
            .iter()
            .zip(marketreturns.iter())
            .map(|(&p, &m)| (p - portfolio_mean) * (m - market_mean))
            .sum::<F>()
            / F::from(portfolioreturns.len() - 1).unwrap();

        let market_variance = marketreturns
            .iter()
            .map(|&m| (m - market_mean) * (m - market_mean))
            .sum::<F>()
            / F::from(marketreturns.len() - 1).unwrap();

        if market_variance > F::zero() {
            Ok(covariance / market_variance)
        } else {
            Ok(F::zero())
        }
    }
}

impl Default for RiskManagementMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Portfolio performance evaluation metrics
#[derive(Debug, Clone)]
pub struct PortfolioMetrics {
    /// Performance attribution results
    pub attribution_results: HashMap<String, f64>,
    /// Risk-adjusted performance metrics
    pub risk_adjusted_metrics: HashMap<String, f64>,
}

impl PortfolioMetrics {
    pub fn new() -> Self {
        Self {
            attribution_results: HashMap::new(),
            risk_adjusted_metrics: HashMap::new(),
        }
    }

    /// Calculate Sharpe Ratio
    pub fn sharpe_ratio<F>(&self, portfolioreturns: &Array1<F>, risk_freerate: F) -> Result<F>
    where
        F: Float + scirs2_core::numeric::FromPrimitive + std::iter::Sum,
    {
        if portfolioreturns.is_empty() {
            return Ok(F::zero());
        }

        let mean_return =
            portfolioreturns.iter().cloned().sum::<F>() / F::from(portfolioreturns.len()).unwrap();
        let excess_return = mean_return - risk_freerate;

        let variance = portfolioreturns
            .iter()
            .map(|&r| (r - mean_return) * (r - mean_return))
            .sum::<F>()
            / F::from(portfolioreturns.len() - 1).unwrap();

        let volatility = variance.sqrt();

        if volatility > F::zero() {
            Ok(excess_return / volatility)
        } else {
            Ok(F::zero())
        }
    }

    /// Calculate Sortino Ratio (downside risk-adjusted return)
    pub fn sortino_ratio<F>(&self, portfolioreturns: &Array1<F>, targetreturn: F) -> Result<F>
    where
        F: Float + scirs2_core::numeric::FromPrimitive + std::iter::Sum,
    {
        if portfolioreturns.is_empty() {
            return Ok(F::zero());
        }

        let mean_return =
            portfolioreturns.iter().cloned().sum::<F>() / F::from(portfolioreturns.len()).unwrap();
        let excess_return = mean_return - targetreturn;

        // Calculate downside deviation
        let downside_variance = portfolioreturns
            .iter()
            .map(|&r| {
                let downside_diff = targetreturn - r;
                if downside_diff > F::zero() {
                    downside_diff * downside_diff
                } else {
                    F::zero()
                }
            })
            .sum::<F>()
            / F::from(portfolioreturns.len() - 1).unwrap();

        let downside_deviation = downside_variance.sqrt();

        if downside_deviation > F::zero() {
            Ok(excess_return / downside_deviation)
        } else {
            Ok(F::zero())
        }
    }

    /// Calculate Information Ratio
    pub fn information_ratio<F>(
        &self,
        portfolioreturns: &Array1<F>,
        benchmark_returns: &Array1<F>,
    ) -> Result<F>
    where
        F: Float + scirs2_core::numeric::FromPrimitive + std::iter::Sum,
    {
        if portfolioreturns.len() != benchmark_returns.len() {
            return Err(MetricsError::InvalidInput(
                "Portfolio and benchmark _returns must have same length".to_string(),
            ));
        }

        // Calculate active _returns
        let active_returns: Vec<F> = portfolioreturns
            .iter()
            .zip(benchmark_returns.iter())
            .map(|(&p, &b)| p - b)
            .collect();

        let mean_active_return =
            active_returns.iter().cloned().sum::<F>() / F::from(active_returns.len()).unwrap();

        // Calculate tracking error (standard deviation of active returns)
        let tracking_error = {
            let variance = active_returns
                .iter()
                .map(|&ar| (ar - mean_active_return) * (ar - mean_active_return))
                .sum::<F>()
                / F::from(active_returns.len() - 1).unwrap();
            variance.sqrt()
        };

        if tracking_error > F::zero() {
            Ok(mean_active_return / tracking_error)
        } else {
            Ok(F::zero())
        }
    }

    /// Calculate Calmar Ratio
    pub fn calmar_ratio<F>(&self, portfolioreturns: &Array1<F>, prices: &Array1<F>) -> Result<F>
    where
        F: Float + scirs2_core::numeric::FromPrimitive + PartialOrd + std::iter::Sum,
    {
        let annualized_return = self.annualized_return(portfolioreturns)?;

        let risk_mgmt = RiskManagementMetrics::new();
        let max_drawdown = risk_mgmt.maximum_drawdown(prices)?;

        if max_drawdown > F::zero() {
            Ok(annualized_return / max_drawdown)
        } else {
            Ok(F::zero())
        }
    }

    /// Calculate annualized return
    fn annualized_return<F>(&self, returns: &Array1<F>) -> Result<F>
    where
        F: Float + scirs2_core::numeric::FromPrimitive + std::iter::Sum,
    {
        if returns.is_empty() {
            return Ok(F::zero());
        }

        let mean_return = returns.iter().cloned().sum::<F>() / F::from(returns.len()).unwrap();
        // Assuming daily returns, multiply by 252 trading days
        Ok(mean_return * F::from(252).unwrap())
    }
}

impl Default for PortfolioMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Credit risk modeling metrics
#[derive(Debug, Clone)]
pub struct CreditRiskMetrics {
    /// Model discrimination metrics
    pub discrimination_metrics: HashMap<String, f64>,
    /// Model calibration metrics
    pub calibration_metrics: HashMap<String, f64>,
    /// Model stability metrics
    pub stability_metrics: HashMap<String, f64>,
}

impl CreditRiskMetrics {
    pub fn new() -> Self {
        Self {
            discrimination_metrics: HashMap::new(),
            calibration_metrics: HashMap::new(),
            stability_metrics: HashMap::new(),
        }
    }

    /// Calculate Gini coefficient for credit model discrimination
    pub fn gini_coefficient<F>(
        &self,
        predicted_scores: &Array1<F>,
        actual_defaults: &Array1<bool>,
    ) -> Result<F>
    where
        F: Float + scirs2_core::numeric::FromPrimitive + PartialOrd,
    {
        if predicted_scores.len() != actual_defaults.len() {
            return Err(MetricsError::InvalidInput(
                "Predicted _scores and actual _defaults must have same length".to_string(),
            ));
        }

        // Create (score, default) pairs and sort by score descending
        let mut pairs: Vec<(F, bool)> = predicted_scores
            .iter()
            .zip(actual_defaults.iter())
            .map(|(&score, &default)| (score, default))
            .collect();
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let total_defaults = actual_defaults.iter().filter(|&&x| x).count();
        let total_non_defaults = actual_defaults.len() - total_defaults;

        if total_defaults == 0 || total_non_defaults == 0 {
            return Ok(F::zero());
        }

        let mut cumulative_defaults = 0;
        let mut _cumulative_non_defaults = 0;
        let mut auc = F::zero();

        for (_, is_default) in pairs {
            if is_default {
                cumulative_defaults += 1;
            } else {
                _cumulative_non_defaults += 1;
                auc = auc + F::from(cumulative_defaults).unwrap();
            }
        }

        let auc_normalized =
            auc / (F::from(total_defaults).unwrap() * F::from(total_non_defaults).unwrap());
        let gini = F::from(2.0).unwrap() * auc_normalized - F::one();

        Ok(gini)
    }

    /// Calculate Kolmogorov-Smirnov statistic
    pub fn kolmogorov_smirnov<F>(
        &self,
        predicted_scores: &Array1<F>,
        actual_defaults: &Array1<bool>,
    ) -> Result<F>
    where
        F: Float + scirs2_core::numeric::FromPrimitive + PartialOrd,
    {
        if predicted_scores.len() != actual_defaults.len() {
            return Err(MetricsError::InvalidInput(
                "Predicted _scores and actual _defaults must have same length".to_string(),
            ));
        }

        // Create (score, default) pairs and sort by score
        let mut pairs: Vec<(F, bool)> = predicted_scores
            .iter()
            .zip(actual_defaults.iter())
            .map(|(&score, &default)| (score, default))
            .collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let total_defaults = actual_defaults.iter().filter(|&&x| x).count();
        let total_non_defaults = actual_defaults.len() - total_defaults;

        if total_defaults == 0 || total_non_defaults == 0 {
            return Ok(F::zero());
        }

        let mut cumulative_defaults = 0;
        let mut cumulative_non_defaults = 0;
        let mut max_ks = F::zero();

        for (_, is_default) in pairs {
            if is_default {
                cumulative_defaults += 1;
            } else {
                cumulative_non_defaults += 1;
            }

            let default_rate =
                F::from(cumulative_defaults).unwrap() / F::from(total_defaults).unwrap();
            let non_default_rate =
                F::from(cumulative_non_defaults).unwrap() / F::from(total_non_defaults).unwrap();

            let ks_stat = (default_rate - non_default_rate).abs();
            if ks_stat > max_ks {
                max_ks = ks_stat;
            }
        }

        Ok(max_ks)
    }

    /// Calculate Population Stability Index (PSI)
    pub fn population_stability_index<F>(
        &self,
        baseline_scores: &Array1<F>,
        current_scores: &Array1<F>,
        num_buckets: usize,
    ) -> Result<F>
    where
        F: Float + scirs2_core::numeric::FromPrimitive + PartialOrd + Clone + std::iter::Sum,
    {
        if num_buckets == 0 {
            return Err(MetricsError::InvalidInput(
                "Number of _buckets must be greater than 0".to_string(),
            ));
        }

        // Create score _buckets based on baseline distribution
        let mut baseline_sorted: Vec<F> = baseline_scores.iter().cloned().collect();
        baseline_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let bucket_size = baseline_sorted.len() / num_buckets;
        let mut bucket_boundaries = Vec::new();

        for i in 1..num_buckets {
            let idx = i * bucket_size;
            if idx < baseline_sorted.len() {
                bucket_boundaries.push(baseline_sorted[idx]);
            }
        }

        // Count observations in each bucket for baseline and current
        let baseline_counts =
            self.count_observations_in_buckets(baseline_scores, &bucket_boundaries);
        let current_counts = self.count_observations_in_buckets(current_scores, &bucket_boundaries);

        // Calculate PSI
        let mut psi = F::zero();
        for i in 0..num_buckets {
            let baseline_pct =
                F::from(baseline_counts[i]).unwrap() / F::from(baseline_scores.len()).unwrap();
            let current_pct =
                F::from(current_counts[i]).unwrap() / F::from(current_scores.len()).unwrap();

            if baseline_pct > F::zero() && current_pct > F::zero() {
                let ratio = current_pct / baseline_pct;
                psi = psi + (current_pct - baseline_pct) * ratio.ln();
            }
        }

        Ok(psi)
    }

    /// Helper method to count observations in buckets
    fn count_observations_in_buckets<F>(&self, scores: &Array1<F>, boundaries: &[F]) -> Vec<usize>
    where
        F: Float + PartialOrd,
    {
        let num_buckets = boundaries.len() + 1;
        let mut counts = vec![0; num_buckets];

        for &score in scores.iter() {
            let mut bucket = 0;
            for (i, &boundary) in boundaries.iter().enumerate() {
                if score >= boundary {
                    bucket = i + 1;
                } else {
                    break;
                }
            }
            if bucket < counts.len() {
                counts[bucket] += 1;
            }
        }

        counts
    }
}

impl Default for CreditRiskMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Market risk assessment metrics
#[derive(Debug, Clone)]
pub struct MarketRiskMetrics;

impl MarketRiskMetrics {
    pub fn new() -> Self {
        Self
    }

    /// Calculate correlation matrix between multiple assets
    pub fn correlation_matrix<F>(&self, returnsmatrix: &Array2<F>) -> Result<Array2<F>>
    where
        F: Float + scirs2_core::numeric::FromPrimitive + std::iter::Sum,
    {
        let (n_periods, n_assets) = returnsmatrix.dim();
        let mut correlation_matrix = Array2::zeros((n_assets, n_assets));

        // Calculate means for each asset
        let means: Vec<F> = (0..n_assets)
            .map(|i| {
                returnsmatrix.column(i).iter().cloned().sum::<F>() / F::from(n_periods).unwrap()
            })
            .collect();

        // Calculate correlation coefficients
        for i in 0..n_assets {
            for j in 0..n_assets {
                if i == j {
                    correlation_matrix[[i, j]] = F::one();
                } else {
                    let correlation = self.calculate_correlation(
                        &returnsmatrix.column(i),
                        &returnsmatrix.column(j),
                        means[i],
                        means[j],
                    )?;
                    correlation_matrix[[i, j]] = correlation;
                }
            }
        }

        Ok(correlation_matrix)
    }

    /// Helper method to calculate correlation between two return series
    fn calculate_correlation<F>(
        &self,
        returns1: &ArrayView1<F>,
        returns2: &ArrayView1<F>,
        mean1: F,
        mean2: F,
    ) -> Result<F>
    where
        F: Float + scirs2_core::numeric::FromPrimitive + std::iter::Sum,
    {
        let n = F::from(returns1.len()).unwrap();

        let covariance = returns1
            .iter()
            .zip(returns2.iter())
            .map(|(&r1, &r2)| (r1 - mean1) * (r2 - mean2))
            .sum::<F>()
            / (n - F::one());

        let var1 = returns1
            .iter()
            .map(|&r| (r - mean1) * (r - mean1))
            .sum::<F>()
            / (n - F::one());

        let var2 = returns2
            .iter()
            .map(|&r| (r - mean2) * (r - mean2))
            .sum::<F>()
            / (n - F::one());

        let std_product = (var1 * var2).sqrt();

        if std_product > F::zero() {
            Ok(covariance / std_product)
        } else {
            Ok(F::zero())
        }
    }
}

impl Default for MarketRiskMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Trading strategy evaluation metrics
#[derive(Debug, Clone)]
pub struct TradingStrategyMetrics;

impl TradingStrategyMetrics {
    pub fn new() -> Self {
        Self
    }

    /// Calculate hit ratio (percentage of profitable trades)
    pub fn hit_ratio<F>(&self, tradereturns: &Array1<F>) -> Result<F>
    where
        F: Float + scirs2_core::numeric::FromPrimitive + PartialOrd,
    {
        if tradereturns.is_empty() {
            return Ok(F::zero());
        }

        let profitable_trades = tradereturns.iter().filter(|&&ret| ret > F::zero()).count();

        Ok(F::from(profitable_trades).unwrap() / F::from(tradereturns.len()).unwrap())
    }

    /// Calculate profit factor (gross profit / gross loss)
    pub fn profit_factor<F>(&self, tradereturns: &Array1<F>) -> Result<F>
    where
        F: Float + scirs2_core::numeric::FromPrimitive + PartialOrd + std::iter::Sum,
    {
        let gross_profit = tradereturns
            .iter()
            .filter(|&&ret| ret > F::zero())
            .cloned()
            .sum::<F>();

        let gross_loss = tradereturns
            .iter()
            .filter(|&&ret| ret < F::zero())
            .cloned()
            .sum::<F>()
            .abs();

        if gross_loss > F::zero() {
            Ok(gross_profit / gross_loss)
        } else if gross_profit > F::zero() {
            Ok(F::infinity())
        } else {
            Ok(F::zero())
        }
    }

    /// Calculate recovery factor
    pub fn recovery_factor<F>(&self, tradereturns: &Array1<F>, prices: &Array1<F>) -> Result<F>
    where
        F: Float + scirs2_core::numeric::FromPrimitive + PartialOrd + std::iter::Sum,
    {
        let total_return = tradereturns.iter().cloned().sum::<F>();

        let risk_mgmt = RiskManagementMetrics::new();
        let max_drawdown = risk_mgmt.maximum_drawdown(prices)?;

        if max_drawdown > F::zero() {
            Ok(total_return / max_drawdown)
        } else {
            Ok(F::zero())
        }
    }
}

impl Default for TradingStrategyMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Fraud detection metrics
#[derive(Debug, Clone)]
pub struct FraudDetectionMetrics;

impl FraudDetectionMetrics {
    pub fn new() -> Self {
        Self
    }

    /// Calculate fraud detection rate
    pub fn fraud_detection_rate(
        &self,
        true_fraud: &Array1<bool>,
        predicted_fraud: &Array1<bool>,
    ) -> Result<f64> {
        if true_fraud.len() != predicted_fraud.len() {
            return Err(MetricsError::InvalidInput(
                "True and predicted arrays must have same length".to_string(),
            ));
        }

        let true_positives = true_fraud
            .iter()
            .zip(predicted_fraud.iter())
            .filter(|(&actual, &predicted)| actual && predicted)
            .count();

        let total_fraud = true_fraud.iter().filter(|&&x| x).count();

        if total_fraud > 0 {
            Ok(true_positives as f64 / total_fraud as f64)
        } else {
            Ok(0.0)
        }
    }

    /// Calculate false positive rate
    pub fn false_positive_rate(
        &self,
        true_fraud: &Array1<bool>,
        predicted_fraud: &Array1<bool>,
    ) -> Result<f64> {
        if true_fraud.len() != predicted_fraud.len() {
            return Err(MetricsError::InvalidInput(
                "True and predicted arrays must have same length".to_string(),
            ));
        }

        let false_positives = true_fraud
            .iter()
            .zip(predicted_fraud.iter())
            .filter(|(&actual, &predicted)| !actual && predicted)
            .count();

        let total_legitimate = true_fraud.iter().filter(|&&x| !x).count();

        if total_legitimate > 0 {
            Ok(false_positives as f64 / total_legitimate as f64)
        } else {
            Ok(0.0)
        }
    }
}

impl Default for FraudDetectionMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Regulatory compliance metrics (Basel III, etc.)
#[derive(Debug, Clone)]
pub struct RegulatoryMetrics;

impl RegulatoryMetrics {
    pub fn new() -> Self {
        Self
    }

    /// Calculate regulatory capital ratio
    pub fn capital_ratio<F>(&self, tier1_capital: F, risk_weightedassets: F) -> Result<F>
    where
        F: Float,
    {
        if risk_weightedassets > F::zero() {
            Ok(tier1_capital / risk_weightedassets)
        } else {
            Err(MetricsError::InvalidInput(
                "Risk-weighted _assets must be greater than zero".to_string(),
            ))
        }
    }

    /// Calculate leverage ratio
    pub fn leverage_ratio<F>(&self, tier1_capital: F, totalexposure: F) -> Result<F>
    where
        F: Float,
    {
        if totalexposure > F::zero() {
            Ok(tier1_capital / totalexposure)
        } else {
            Err(MetricsError::InvalidInput(
                "Total _exposure must be greater than zero".to_string(),
            ))
        }
    }

    /// Calculate Liquidity Coverage Ratio (LCR)
    pub fn liquidity_coverage_ratio<F>(
        &self,
        high_quality_liquid_assets: F,
        net_cash_outflows: F,
    ) -> Result<F>
    where
        F: Float,
    {
        if net_cash_outflows > F::zero() {
            Ok(high_quality_liquid_assets / net_cash_outflows)
        } else {
            Err(MetricsError::InvalidInput(
                "Net cash _outflows must be greater than zero".to_string(),
            ))
        }
    }
}

impl Default for RegulatoryMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// ESG (Environmental, Social, Governance) scoring metrics
#[derive(Debug, Clone)]
pub struct ESGMetrics;

impl ESGMetrics {
    pub fn new() -> Self {
        Self
    }

    /// Calculate composite ESG score
    pub fn composite_esg_score<F>(
        &self,
        environmental_score: F,
        social_score: F,
        governance_score: F,
        weights: Option<(F, F, F)>,
    ) -> Result<F>
    where
        F: Float + scirs2_core::numeric::FromPrimitive,
    {
        let (env_weight, social_weight, gov_weight) = weights.unwrap_or((
            F::from(1.0).unwrap() / F::from(3.0).unwrap(),
            F::from(1.0).unwrap() / F::from(3.0).unwrap(),
            F::from(1.0).unwrap() / F::from(3.0).unwrap(),
        ));

        Ok(environmental_score * env_weight
            + social_score * social_weight
            + governance_score * gov_weight)
    }

    /// Calculate carbon intensity metric
    pub fn carbon_intensity<F>(&self, carbonemissions: F, revenue: F) -> Result<F>
    where
        F: Float,
    {
        if revenue > F::zero() {
            Ok(carbonemissions / revenue)
        } else {
            Err(MetricsError::InvalidInput(
                "Revenue must be greater than zero".to_string(),
            ))
        }
    }
}

impl Default for ESGMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_financial_suite_creation() {
        let suite = FinancialSuite::new();
        assert_eq!(
            suite.domain_name(),
            "Financial Modeling & Quantitative Finance"
        );

        let metrics = suite.available_metrics();
        assert!(metrics.contains(&"value_at_risk"));
        assert!(metrics.contains(&"sharpe_ratio"));
        assert!(metrics.contains(&"gini_coefficient"));
    }

    #[test]
    fn test_sharpe_ratio() {
        let portfolio = PortfolioMetrics::new();
        let returns = array![0.01, 0.02, -0.01, 0.03, 0.0];
        let risk_freerate = 0.005;

        let sharpe = portfolio.sharpe_ratio(&returns, risk_freerate).unwrap();
        assert!(sharpe.is_finite());
    }

    #[test]
    fn test_historical_var() {
        let risk_mgmt = RiskManagementMetrics::new();
        let returns = array![-0.05, -0.02, 0.01, 0.03, -0.01, 0.02, -0.03];
        let confidencelevel = 0.95;

        let var = risk_mgmt
            .historical_var(&returns, confidencelevel, 1)
            .unwrap();
        assert!(var >= 0.0);
    }

    #[test]
    fn test_gini_coefficient() {
        let credit = CreditRiskMetrics::new();
        let scores = array![0.8, 0.6, 0.9, 0.3, 0.7];
        let defaults = array![true, false, true, false, false];

        let gini = credit.gini_coefficient(&scores, &defaults).unwrap();
        assert!((-1.0..=1.0).contains(&gini));
    }

    #[test]
    fn test_hit_ratio() {
        let trading = TradingStrategyMetrics::new();
        let tradereturns = array![0.02, -0.01, 0.03, -0.005, 0.01];

        let hit_ratio = trading.hit_ratio(&tradereturns).unwrap();
        assert!((0.0..=1.0).contains(&hit_ratio));
        assert_eq!(hit_ratio, 0.6); // 3 out of 5 profitable trades
    }

    #[test]
    fn test_capital_ratio() {
        let regulatory = RegulatoryMetrics::new();
        let tier1_capital = 100.0;
        let risk_weightedassets = 800.0;

        let ratio = regulatory
            .capital_ratio(tier1_capital, risk_weightedassets)
            .unwrap();
        assert_eq!(ratio, 0.125); // 12.5%
    }

    #[test]
    fn test_composite_esg_score() {
        let esg = ESGMetrics::new();
        let environmental = 0.8;
        let social = 0.7;
        let governance = 0.9;

        let composite = esg
            .composite_esg_score(environmental, social, governance, None)
            .unwrap();
        assert!((0.0..=1.0).contains(&composite));
    }
}
