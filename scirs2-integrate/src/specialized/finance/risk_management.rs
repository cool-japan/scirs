//! Portfolio-level risk analytics facade
//!
//! This module aggregates
//! [`risk::var::HistoricalVaR`](crate::specialized::finance::risk::var::HistoricalVaR),
//! [`risk::greeks::Greeks`](crate::specialized::finance::risk::greeks::Greeks)
//! (Black-Scholes sensitivities), and
//! [`monte_carlo::mc_portfolio_var`](crate::specialized::finance::monte_carlo::mc_portfolio_var)
//! into a single portfolio-level risk report ([`PortfolioRiskMetrics`])
//! computed by [`RiskAnalyzer`].
//!
//! `StressScenario` (portfolio-shock revaluation) is intentionally NOT
//! implemented here — see `TODO.md` "Proposed follow-ups" for why.

use crate::error::{IntegrateError, IntegrateResult};
use crate::specialized::finance::monte_carlo::mc_portfolio_var;
use crate::specialized::finance::risk::greeks::Greeks;
use crate::specialized::finance::risk::var::HistoricalVaR;
use crate::specialized::finance::types::OptionType;
use scirs2_core::ndarray::{Array1, Array2};

// ============================================================
// PortfolioPosition
// ============================================================

/// A single option position within a portfolio, described by the inputs
/// needed to compute Black-Scholes Greeks via
/// [`Greeks::black_scholes`].
#[derive(Debug, Clone)]
pub struct PortfolioPosition {
    /// Free-form label identifying this position (e.g. ticker + strike).
    pub label: String,
    /// Underlying spot price.
    pub spot: f64,
    /// Strike price.
    pub strike: f64,
    /// Risk-free rate.
    pub rate: f64,
    /// Continuous dividend yield.
    pub dividend: f64,
    /// Volatility (annualized).
    pub volatility: f64,
    /// Time to expiry, in years.
    pub time_to_expiry: f64,
    /// Call or put.
    pub option_type: OptionType,
    /// Signed position size (positive = long, negative = short).
    pub quantity: f64,
}

// ============================================================
// PortfolioRiskMetrics
// ============================================================

/// Aggregated portfolio-level risk metrics: historical and Monte Carlo VaR
/// alongside net position Greeks.
#[derive(Debug, Clone)]
pub struct PortfolioRiskMetrics {
    /// Historical (empirical-distribution) Value at Risk.
    pub historical_var: f64,
    /// Historical Conditional VaR (Expected Shortfall).
    pub historical_cvar: f64,
    /// Confidence level used for both VaR computations (e.g. 0.95, 0.99).
    pub confidence_level: f64,
    /// Monte Carlo (correlated log-normal scenario) portfolio VaR.
    pub monte_carlo_var: f64,
    /// Monte Carlo Conditional VaR.
    pub monte_carlo_cvar: f64,
    /// Net delta across all positions (quantity-weighted sum).
    pub net_delta: f64,
    /// Net gamma across all positions.
    pub net_gamma: f64,
    /// Net vega across all positions.
    pub net_vega: f64,
    /// Net theta across all positions.
    pub net_theta: f64,
    /// Net rho across all positions.
    pub net_rho: f64,
    /// Per-position label and raw (unscaled) Greeks.
    pub per_position_greeks: Vec<(String, Greeks)>,
}

// ============================================================
// RiskAnalyzer
// ============================================================

/// Computes [`PortfolioRiskMetrics`] from historical returns, a Monte Carlo
/// covariance model, and a set of option positions.
#[derive(Debug, Clone, Copy)]
pub struct RiskAnalyzer {
    /// Confidence level applied to both the historical and Monte Carlo VaR
    /// computations (must be in `(0, 1)`).
    pub confidence_level: f64,
}

impl RiskAnalyzer {
    /// Create a new analyzer for the given confidence level.
    pub fn new(confidence_level: f64) -> IntegrateResult<Self> {
        if confidence_level <= 0.0 || confidence_level >= 1.0 {
            return Err(IntegrateError::ValueError(
                "Confidence level must be between 0 and 1".to_string(),
            ));
        }
        Ok(Self { confidence_level })
    }

    /// Aggregate historical VaR (via [`HistoricalVaR`]), Monte Carlo
    /// portfolio VaR (via [`mc_portfolio_var`]), and net position Greeks
    /// (via [`Greeks::black_scholes`]) into one report.
    ///
    /// # Arguments
    /// - `historical_returns`: historical daily returns for the historical
    ///   VaR leg.
    /// - `horizon_days`: horizon (in days) to scale the historical VaR to.
    /// - `initial_prices`/`weights`/`mu`/`cov_matrix`: portfolio composition
    ///   and covariance model for the Monte Carlo VaR leg (see
    ///   [`mc_portfolio_var`]).
    /// - `mc_horizon_years`/`n_scenarios`/`seed`: Monte Carlo simulation
    ///   controls.
    /// - `positions`: option positions to aggregate net Greeks over.
    #[allow(clippy::too_many_arguments)]
    pub fn analyze(
        &self,
        historical_returns: Vec<f64>,
        horizon_days: usize,
        initial_prices: &Array1<f64>,
        weights: &Array1<f64>,
        mu: &Array1<f64>,
        cov_matrix: &Array2<f64>,
        mc_horizon_years: f64,
        n_scenarios: usize,
        seed: u64,
        positions: &[PortfolioPosition],
    ) -> IntegrateResult<PortfolioRiskMetrics> {
        let hist_var_calc = HistoricalVaR::new(historical_returns, self.confidence_level)?;
        let hist_result = hist_var_calc.calculate(horizon_days);

        let (mc_var, mc_cvar) = mc_portfolio_var(
            initial_prices,
            weights,
            mu,
            cov_matrix,
            mc_horizon_years,
            self.confidence_level,
            n_scenarios,
            seed,
        )?;

        let mut per_position_greeks = Vec::with_capacity(positions.len());
        let (mut net_delta, mut net_gamma, mut net_vega, mut net_theta, mut net_rho) =
            (0.0, 0.0, 0.0, 0.0, 0.0);
        for pos in positions {
            let g = Greeks::black_scholes(
                pos.spot,
                pos.strike,
                pos.rate,
                pos.dividend,
                pos.volatility,
                pos.time_to_expiry,
                pos.option_type,
            );
            net_delta += g.delta * pos.quantity;
            net_gamma += g.gamma * pos.quantity;
            net_vega += g.vega * pos.quantity;
            net_theta += g.theta * pos.quantity;
            net_rho += g.rho * pos.quantity;
            per_position_greeks.push((pos.label.clone(), g));
        }

        Ok(PortfolioRiskMetrics {
            historical_var: hist_result.var,
            historical_cvar: hist_result.cvar,
            confidence_level: self.confidence_level,
            monte_carlo_var: mc_var,
            monte_carlo_cvar: mc_cvar,
            net_delta,
            net_gamma,
            net_vega,
            net_theta,
            net_rho,
            per_position_greeks,
        })
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_risk_analyzer_matches_direct_primitive_calls() {
        let historical_returns = vec![
            0.01, 0.02, -0.01, 0.015, -0.02, 0.005, -0.015, 0.01, -0.025, 0.02,
        ];
        let confidence = 0.95;
        let horizon_days = 1;

        let prices = array![100.0, 200.0];
        let weights = array![0.5, 0.5];
        let mu = array![0.08, 0.10];
        let cov = Array2::from_shape_vec((2, 2), vec![0.04, 0.01, 0.01, 0.09]).expect("cov shape");
        let mc_horizon = 1.0 / 252.0;
        let n_scenarios = 10_000;
        let seed = 42;

        let position = PortfolioPosition {
            label: "AAPL_CALL".to_string(),
            spot: 100.0,
            strike: 100.0,
            rate: 0.05,
            dividend: 0.0,
            volatility: 0.2,
            time_to_expiry: 1.0,
            option_type: OptionType::Call,
            quantity: 10.0,
        };

        let analyzer = RiskAnalyzer::new(confidence).expect("analyzer construction failed");
        let metrics = analyzer
            .analyze(
                historical_returns.clone(),
                horizon_days,
                &prices,
                &weights,
                &mu,
                &cov,
                mc_horizon,
                n_scenarios,
                seed,
                std::slice::from_ref(&position),
            )
            .expect("analyze failed");

        // Independently compute each component: same seed => bit-identical
        // Monte Carlo result; the historical VaR and Greeks legs are
        // deterministic closed forms.
        let direct_hist = HistoricalVaR::new(historical_returns, confidence)
            .expect("historical var construction failed")
            .calculate(horizon_days);
        let (direct_mc_var, direct_mc_cvar) = mc_portfolio_var(
            &prices,
            &weights,
            &mu,
            &cov,
            mc_horizon,
            confidence,
            n_scenarios,
            seed,
        )
        .expect("direct mc var failed");
        let direct_greeks = Greeks::black_scholes(
            position.spot,
            position.strike,
            position.rate,
            position.dividend,
            position.volatility,
            position.time_to_expiry,
            position.option_type,
        );

        assert_relative_eq!(metrics.historical_var, direct_hist.var, epsilon = 1e-12);
        assert_relative_eq!(metrics.historical_cvar, direct_hist.cvar, epsilon = 1e-12);
        assert_relative_eq!(metrics.monte_carlo_var, direct_mc_var, epsilon = 1e-12);
        assert_relative_eq!(metrics.monte_carlo_cvar, direct_mc_cvar, epsilon = 1e-12);
        assert_relative_eq!(
            metrics.net_delta,
            direct_greeks.delta * 10.0,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            metrics.net_gamma,
            direct_greeks.gamma * 10.0,
            epsilon = 1e-12
        );
        assert_eq!(metrics.per_position_greeks.len(), 1);
        assert_eq!(metrics.per_position_greeks[0].0, "AAPL_CALL");
    }

    #[test]
    fn test_risk_analyzer_rejects_invalid_confidence() {
        assert!(RiskAnalyzer::new(1.5).is_err());
        assert!(RiskAnalyzer::new(0.0).is_err());
        assert!(RiskAnalyzer::new(0.95).is_ok());
    }
}
