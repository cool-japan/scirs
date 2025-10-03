//! Value at Risk (VaR) and risk metrics
//!
//! This module implements various VaR calculation methods for portfolio risk management.

use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::random::{Normal, Rng};

/// VaR calculation result
#[derive(Debug, Clone)]
pub struct VaRResult {
    /// Value at Risk at specified confidence level
    pub var: f64,
    /// Conditional Value at Risk (Expected Shortfall)
    pub cvar: f64,
    /// Confidence level (e.g., 0.95, 0.99)
    pub confidence_level: f64,
    /// Time horizon in days
    pub horizon_days: usize,
}

/// Historical VaR calculator using empirical distribution
pub struct HistoricalVaR {
    /// Historical returns (daily)
    returns: Vec<f64>,
    /// Confidence level
    confidence_level: f64,
}

impl HistoricalVaR {
    /// Create a new historical VaR calculator
    pub fn new(returns: Vec<f64>, confidence_level: f64) -> IntegrateResult<Self> {
        if confidence_level <= 0.0 || confidence_level >= 1.0 {
            return Err(IntegrateError::ValueError(
                "Confidence level must be between 0 and 1".to_string(),
            ));
        }

        if returns.is_empty() {
            return Err(IntegrateError::ValueError(
                "Returns data cannot be empty".to_string(),
            ));
        }

        Ok(Self {
            returns,
            confidence_level,
        })
    }

    /// Calculate VaR and CVaR for a given horizon
    pub fn calculate(&self, horizon_days: usize) -> VaRResult {
        let mut sorted_returns = self.returns.clone();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Scale returns to horizon
        let horizon_factor = (horizon_days as f64).sqrt();
        let scaled_returns: Vec<f64> = sorted_returns.iter().map(|r| r * horizon_factor).collect();

        // Calculate VaR as the percentile
        let alpha = 1.0 - self.confidence_level;
        let var_index = (alpha * scaled_returns.len() as f64).floor() as usize;
        let var = -scaled_returns[var_index.min(scaled_returns.len() - 1)];

        // Calculate CVaR as average of losses beyond VaR
        let tail_losses: Vec<f64> = scaled_returns.iter().take(var_index + 1).copied().collect();

        let cvar = if !tail_losses.is_empty() {
            -tail_losses.iter().sum::<f64>() / tail_losses.len() as f64
        } else {
            var
        };

        VaRResult {
            var,
            cvar,
            confidence_level: self.confidence_level,
            horizon_days,
        }
    }
}

/// Parametric VaR calculator using variance-covariance method
pub struct ParametricVaR {
    /// Portfolio mean return
    mean_return: f64,
    /// Portfolio volatility (standard deviation)
    volatility: f64,
    /// Confidence level
    confidence_level: f64,
}

impl ParametricVaR {
    /// Create a new parametric VaR calculator
    pub fn new(mean_return: f64, volatility: f64, confidence_level: f64) -> IntegrateResult<Self> {
        if confidence_level <= 0.0 || confidence_level >= 1.0 {
            return Err(IntegrateError::ValueError(
                "Confidence level must be between 0 and 1".to_string(),
            ));
        }

        if volatility < 0.0 {
            return Err(IntegrateError::ValueError(
                "Volatility must be non-negative".to_string(),
            ));
        }

        Ok(Self {
            mean_return,
            volatility,
            confidence_level,
        })
    }

    /// Calculate VaR assuming normal distribution
    pub fn calculate(&self, horizon_days: usize) -> VaRResult {
        let horizon_factor = (horizon_days as f64).sqrt();
        let alpha = 1.0 - self.confidence_level;

        // Z-score for confidence level (normal distribution)
        let z_score = inverse_normal_cdf(alpha);

        // VaR = -(μ - z*σ) for loss
        let var =
            -(self.mean_return * horizon_days as f64 + z_score * self.volatility * horizon_factor);

        // CVaR = -(μ - σ * φ(z) / α) where φ is standard normal PDF
        let phi_z = (1.0 / (2.0 * std::f64::consts::PI).sqrt()) * (-0.5 * z_score * z_score).exp();
        let cvar = -(self.mean_return * horizon_days as f64
            - self.volatility * horizon_factor * phi_z / alpha);

        VaRResult {
            var,
            cvar,
            confidence_level: self.confidence_level,
            horizon_days,
        }
    }
}

/// Monte Carlo VaR calculator with scenario generation
pub struct MonteCarloVaR {
    /// Portfolio mean return
    mean_return: f64,
    /// Portfolio volatility
    volatility: f64,
    /// Number of simulation paths
    n_simulations: usize,
    /// Confidence level
    confidence_level: f64,
}

impl MonteCarloVaR {
    /// Create a new Monte Carlo VaR calculator
    pub fn new(
        mean_return: f64,
        volatility: f64,
        n_simulations: usize,
        confidence_level: f64,
    ) -> IntegrateResult<Self> {
        if confidence_level <= 0.0 || confidence_level >= 1.0 {
            return Err(IntegrateError::ValueError(
                "Confidence level must be between 0 and 1".to_string(),
            ));
        }

        if n_simulations == 0 {
            return Err(IntegrateError::ValueError(
                "Number of simulations must be positive".to_string(),
            ));
        }

        Ok(Self {
            mean_return,
            volatility,
            n_simulations,
            confidence_level,
        })
    }

    /// Calculate VaR using Monte Carlo simulation
    pub fn calculate(&self, horizon_days: usize) -> VaRResult {
        let mut rng = scirs2_core::random::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        let dt = 1.0; // Daily steps
        let n_steps = horizon_days;

        // Generate scenarios
        let mut portfolio_returns = Vec::with_capacity(self.n_simulations);

        for _ in 0..self.n_simulations {
            let mut cumulative_return = 0.0;

            for _ in 0..n_steps {
                let z: f64 = rng.sample(normal);
                let daily_return = self.mean_return * dt + self.volatility * dt.sqrt() * z;
                cumulative_return += daily_return;
            }

            portfolio_returns.push(cumulative_return);
        }

        // Sort returns
        portfolio_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Calculate VaR
        let alpha = 1.0 - self.confidence_level;
        let var_index = (alpha * portfolio_returns.len() as f64).floor() as usize;
        let var = -portfolio_returns[var_index.min(portfolio_returns.len() - 1)];

        // Calculate CVaR
        let tail_losses: Vec<f64> = portfolio_returns
            .iter()
            .take(var_index + 1)
            .copied()
            .collect();

        let cvar = if !tail_losses.is_empty() {
            -tail_losses.iter().sum::<f64>() / tail_losses.len() as f64
        } else {
            var
        };

        VaRResult {
            var,
            cvar,
            confidence_level: self.confidence_level,
            horizon_days,
        }
    }
}

/// Approximate inverse normal CDF (for parametric VaR)
fn inverse_normal_cdf(p: f64) -> f64 {
    // Beasley-Springer-Moro algorithm approximation
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383_577_518_672_69e2,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];

    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];

    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];

    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }

    if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    }

    let q = (-2.0 * (1.0 - p).ln()).sqrt();
    -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_historical_var() {
        // Generate sample returns
        let returns = vec![
            0.01, 0.02, -0.01, 0.015, -0.02, 0.005, -0.015, 0.01, -0.025, 0.02,
        ];

        let var_calc = HistoricalVaR::new(returns, 0.95).unwrap();
        let result = var_calc.calculate(1);

        assert!(result.var > 0.0, "VaR should be positive");
        assert!(result.cvar >= result.var, "CVaR should be >= VaR");
        assert_eq!(result.confidence_level, 0.95);
    }

    #[test]
    fn test_parametric_var() {
        let var_calc = ParametricVaR::new(0.0005, 0.02, 0.95).unwrap();
        let result = var_calc.calculate(1);

        // For 95% confidence with σ=0.02, VaR should be around 0.0329 (1.645*0.02)
        assert!(
            result.var > 0.025 && result.var < 0.04,
            "VaR: {}",
            result.var
        );
        assert!(result.cvar >= result.var, "CVaR should be >= VaR");
    }

    #[test]
    fn test_monte_carlo_var() {
        let var_calc = MonteCarloVaR::new(0.0005, 0.02, 10000, 0.95).unwrap();
        let result = var_calc.calculate(1);

        // Should be similar to parametric VaR
        assert!(
            result.var > 0.02 && result.var < 0.05,
            "VaR: {}",
            result.var
        );
        assert!(result.cvar >= result.var, "CVaR should be >= VaR");
    }

    #[test]
    fn test_var_multi_day_horizon() {
        let var_calc = ParametricVaR::new(0.0, 0.01, 0.95).unwrap();

        let var_1day = var_calc.calculate(1);
        let var_10day = var_calc.calculate(10);

        // 10-day VaR should be approximately sqrt(10) times 1-day VaR
        let ratio = var_10day.var / var_1day.var;
        assert!(
            ratio > 3.0 && ratio < 3.5,
            "10-day scaling ratio: {}",
            ratio
        );
    }

    #[test]
    fn test_cvar_exceeds_var() {
        let returns = vec![
            -0.05, -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05,
        ];

        let var_calc = HistoricalVaR::new(returns, 0.90).unwrap();
        let result = var_calc.calculate(1);

        // CVaR (Expected Shortfall) should always be >= VaR
        assert!(
            result.cvar >= result.var,
            "CVaR ({}) should be >= VaR ({})",
            result.cvar,
            result.var
        );
    }

    #[test]
    fn test_higher_confidence_higher_var() {
        let var_95 = ParametricVaR::new(0.0, 0.01, 0.95).unwrap();
        let var_99 = ParametricVaR::new(0.0, 0.01, 0.99).unwrap();

        let result_95 = var_95.calculate(1);
        let result_99 = var_99.calculate(1);

        // 99% VaR should be higher than 95% VaR
        assert!(
            result_99.var > result_95.var,
            "99% VaR ({}) should be > 95% VaR ({})",
            result_99.var,
            result_95.var
        );
    }
}
