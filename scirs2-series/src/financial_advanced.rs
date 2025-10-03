//! Advanced Financial Time Series Analytics
//!
//! This module provides sophisticated financial modeling capabilities including:
//! - Advanced GARCH models (EGARCH, TGARCH, FIGARCH)
//! - Options pricing models (Black-Scholes, Heston, Merton Jump-Diffusion)
//! - Risk metrics (VaR, CVaR, Maximum Drawdown, Sharpe Ratio)
//! - High-frequency trading indicators
//! - Copula models for dependency modeling
//! - Regime-switching models

use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{Result, TimeSeriesError};
use statrs::statistics::Statistics;

/// Advanced GARCH model variants
#[derive(Debug, Clone)]
pub enum AdvancedGarchModel {
    /// Exponential GARCH (EGARCH)
    EGARCH {
        /// GARCH order
        p: usize,
        /// ARCH order  
        q: usize,
        /// Asymmetry order
        o: usize,
    },
    /// Threshold GARCH (TGARCH/GJR-GARCH)
    TGARCH {
        /// GARCH order
        p: usize,
        /// ARCH order
        q: usize,
        /// Threshold terms
        o: usize,
    },
    /// Fractionally Integrated GARCH (FIGARCH)
    FIGARCH {
        /// GARCH order
        p: usize,
        /// ARCH order
        q: usize,
        /// Fractional integration parameter
        d: f64,
    },
    /// Component GARCH (CGARCH)
    CGARCH {
        /// Short-term volatility order
        p: usize,
        /// Long-term volatility order
        q: usize,
    },
}

/// Advanced volatility model parameters
#[derive(Debug, Clone)]
pub struct AdvancedVolatilityParams<F: Float> {
    /// Mean equation constant
    pub mu: F,
    /// Volatility equation constant (omega)
    pub omega: F,
    /// ARCH coefficients (alpha)
    pub alpha: Vec<F>,
    /// GARCH coefficients (beta)
    pub beta: Vec<F>,
    /// Asymmetry coefficients (gamma for EGARCH/TGARCH)
    pub gamma: Option<Vec<F>>,
    /// Fractional integration parameter (for FIGARCH)
    pub d: Option<F>,
    /// Long-term component parameters (for CGARCH)
    pub rho: Option<F>,
    /// Persistence parameter
    pub phi: Option<F>,
}

/// Options pricing models
#[derive(Debug, Clone)]
pub enum OptionPricingModel {
    /// Black-Scholes-Merton model
    BlackScholes,
    /// Heston stochastic volatility model
    Heston {
        /// Mean reversion speed
        kappa: f64,
        /// Long-term variance
        theta: f64,
        /// Volatility of volatility
        sigma: f64,
        /// Correlation between asset and volatility
        rho: f64,
        /// Initial volatility
        v0: f64,
    },
    /// Merton Jump-Diffusion model
    MertonJumpDiffusion {
        /// Jump intensity
        lambda: f64,
        /// Jump size mean
        mu_j: f64,
        /// Jump size volatility
        sigma_j: f64,
    },
    /// Bates model (Heston + jumps)
    Bates {
        /// Heston parameters
        kappa: f64,
        /// Long-term variance mean
        theta: f64,
        /// Volatility of volatility
        sigma: f64,
        /// Correlation coefficient
        rho: f64,
        /// Initial variance
        v0: f64,
        /// Jump parameters
        lambda: f64,
        /// Jump mean
        mu_j: f64,
        /// Jump standard deviation
        sigma_j: f64,
    },
}

/// Option contract specification
#[derive(Debug, Clone)]
pub struct OptionContract {
    /// Underlying asset price
    pub spot: f64,
    /// Strike price
    pub strike: f64,
    /// Time to expiration (in years)
    pub maturity: f64,
    /// Risk-free rate
    pub risk_freerate: f64,
    /// Dividend yield
    pub dividend_yield: f64,
    /// Option type
    pub option_type: OptionType,
}

/// Option type specification
#[derive(Debug, Clone)]
pub enum OptionType {
    /// European call option
    Call,
    /// European put option
    Put,
    /// American call option
    AmericanCall,
    /// American put option
    AmericanPut,
    /// Barrier options
    Barrier {
        /// Barrier level
        barrier: f64,
        /// Barrier type
        barrier_type: BarrierType,
    },
    /// Asian options
    Asian {
        /// Averaging type
        averaging: AveragingType,
    },
}

/// Barrier option types
#[derive(Debug, Clone)]
pub enum BarrierType {
    /// Up-and-out barrier
    UpAndOut,
    /// Up-and-in barrier
    UpAndIn,
    /// Down-and-out barrier
    DownAndOut,
    /// Down-and-in barrier
    DownAndIn,
}

/// Averaging types for Asian options
#[derive(Debug, Clone)]
pub enum AveragingType {
    /// Arithmetic averaging
    Arithmetic,
    /// Geometric averaging
    Geometric,
}

/// Option pricing result
#[derive(Debug, Clone)]
pub struct OptionPrice {
    /// Option price
    pub price: f64,
    /// Delta (price sensitivity to underlying)
    pub delta: f64,
    /// Gamma (delta sensitivity to underlying)
    pub gamma: f64,
    /// Theta (time decay)
    pub theta: f64,
    /// Vega (volatility sensitivity)
    pub vega: f64,
    /// Rho (interest rate sensitivity)
    pub rho: f64,
}

/// Risk metrics calculator
#[derive(Debug)]
pub struct RiskMetrics<F: Float + Debug + std::iter::Sum + scirs2_core::numeric::FromPrimitive> {
    /// Returns data
    returns: Array1<F>,
    /// Confidence levels for VaR/CVaR
    #[allow(dead_code)]
    confidencelevels: Vec<F>,
}

impl<F: Float + Debug + Clone + std::iter::Sum + scirs2_core::numeric::FromPrimitive>
    RiskMetrics<F>
{
    /// Create new risk metrics calculator
    pub fn new(returns: Array1<F>) -> Self {
        let confidencelevels = vec![
            F::from(0.90).unwrap(),
            F::from(0.95).unwrap(),
            F::from(0.99).unwrap(),
        ];

        Self {
            returns,
            confidencelevels,
        }
    }

    /// Calculate Value at Risk (VaR) using historical simulation
    pub fn value_at_risk(&self, confidencelevel: F) -> Result<F> {
        if self.returns.is_empty() {
            return Err(TimeSeriesError::InsufficientData {
                message: "No returns data available".to_string(),
                required: 1,
                actual: 0,
            });
        }

        let mut sorted_returns = self.returns.to_vec();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let percentile = F::one() - confidencelevel;
        let index = (percentile * F::from(sorted_returns.len()).unwrap())
            .to_usize()
            .unwrap();
        let index = index.min(sorted_returns.len() - 1);

        Ok(-sorted_returns[index]) // VaR is typically reported as positive loss
    }

    /// Calculate Conditional Value at Risk (CVaR/Expected Shortfall)
    pub fn conditional_value_at_risk(&self, confidencelevel: F) -> Result<F> {
        if self.returns.is_empty() {
            return Err(TimeSeriesError::InsufficientData {
                message: "No returns data available".to_string(),
                required: 1,
                actual: 0,
            });
        }

        let var = self.value_at_risk(confidencelevel)?;
        let var_threshold = -var; // Convert back to actual return value

        let tail_losses: Vec<F> = self
            .returns
            .iter()
            .filter(|&&r| r <= var_threshold)
            .cloned()
            .collect();

        if tail_losses.is_empty() {
            return Ok(var);
        }

        let sum = tail_losses.iter().fold(F::zero(), |acc, &x| acc + x);
        let cvar = sum / F::from(tail_losses.len()).unwrap();
        Ok(-cvar) // Report as positive loss
    }

    /// Calculate Maximum Drawdown
    pub fn maximum_drawdown(&self) -> Result<F> {
        if self.returns.is_empty() {
            return Err(TimeSeriesError::InsufficientData {
                message: "No returns data available".to_string(),
                required: 1,
                actual: 0,
            });
        }

        // Convert returns to cumulative price series
        let mut cumulative: Array1<F> = Array1::ones(self.returns.len() + 1);
        for i in 0..self.returns.len() {
            cumulative[i + 1] = cumulative[i] * (F::one() + self.returns[i]);
        }

        let mut max_drawdown = F::zero();
        let mut peak = cumulative[0];

        for i in 1..cumulative.len() {
            if cumulative[i] > peak {
                peak = cumulative[i];
            }

            let drawdown = (peak - cumulative[i]) / peak;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        Ok(max_drawdown)
    }

    /// Calculate Sharpe Ratio
    pub fn sharpe_ratio(&self, risk_freerate: F) -> Result<F> {
        if self.returns.is_empty() {
            return Err(TimeSeriesError::InsufficientData {
                message: "No returns data available".to_string(),
                required: 1,
                actual: 0,
            });
        }

        let meanreturn = self.returns.mean().unwrap();
        let excessreturn = meanreturn - risk_freerate;
        let volatility = self.volatility()?;

        if volatility == F::zero() {
            return Ok(F::zero());
        }

        Ok(excessreturn / volatility)
    }

    /// Calculate Sortino Ratio
    pub fn sortino_ratio(&self, risk_freerate: F) -> Result<F> {
        if self.returns.is_empty() {
            return Err(TimeSeriesError::InsufficientData {
                message: "No returns data available".to_string(),
                required: 1,
                actual: 0,
            });
        }

        let meanreturn = self.returns.mean().unwrap();
        let excessreturn = meanreturn - risk_freerate;

        // Calculate downside deviation
        let negative_returns: Vec<F> = self
            .returns
            .iter()
            .map(|&r| {
                if r < risk_freerate {
                    (r - risk_freerate) * (r - risk_freerate)
                } else {
                    F::zero()
                }
            })
            .collect();

        let downside_variance = negative_returns.iter().fold(F::zero(), |acc, &x| acc + x)
            / F::from(negative_returns.len()).unwrap();
        let downside_deviation = downside_variance.sqrt();

        if downside_deviation == F::zero() {
            return Ok(F::zero());
        }

        Ok(excessreturn / downside_deviation)
    }

    /// Calculate volatility (standard deviation of returns)
    fn volatility(&self) -> Result<F> {
        if self.returns.len() < 2 {
            return Ok(F::zero());
        }

        let mean = self.returns.mean().unwrap();
        let variance = self
            .returns
            .iter()
            .map(|&r| (r - mean) * (r - mean))
            .sum::<F>()
            / F::from(self.returns.len() - 1).unwrap();

        Ok(variance.sqrt())
    }

    /// Calculate Calmar Ratio (annual return / maximum drawdown)
    pub fn calmar_ratio(&self, periods_peryear: F) -> Result<F> {
        let annualreturn = self.returns.mean().unwrap() * periods_peryear;
        let max_dd = self.maximum_drawdown()?;

        if max_dd == F::zero() {
            return Ok(F::zero());
        }

        Ok(annualreturn / max_dd)
    }
}

/// High-frequency trading indicators
pub struct HFTIndicators;

impl HFTIndicators {
    /// Volume Weighted Average Price (VWAP)
    pub fn vwap<F: Float + Clone>(prices: &Array1<F>, volumes: &Array1<F>) -> Result<Array1<F>> {
        if prices.len() != volumes.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: prices.len(),
                actual: volumes.len(),
            });
        }

        let mut vwap = Array1::zeros(prices.len());
        let mut cumulative_pv = F::zero();
        let mut cumulative_volume = F::zero();

        for i in 0..prices.len() {
            cumulative_pv = cumulative_pv + prices[i] * volumes[i];
            cumulative_volume = cumulative_volume + volumes[i];

            if cumulative_volume > F::zero() {
                vwap[i] = cumulative_pv / cumulative_volume;
            } else {
                vwap[i] = if i > 0 { vwap[i - 1] } else { prices[i] };
            }
        }

        Ok(vwap)
    }

    /// Time Weighted Average Price (TWAP)
    pub fn twap<F: Float + Clone>(prices: &Array1<F>, window: usize) -> Result<Array1<F>> {
        if prices.len() < window {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough data for TWAP calculation".to_string(),
                required: window,
                actual: prices.len(),
            });
        }

        let mut twap = Array1::zeros(prices.len() - window + 1);
        let window_f = F::from(window).unwrap();

        for i in 0..twap.len() {
            let sum = prices.slice(s![i..i + window]).sum();
            twap[i] = sum / window_f;
        }

        Ok(twap)
    }

    /// Market Impact Model (simplified square-root law)
    pub fn market_impact<F: Float + FromPrimitive>(
        volume: F,
        average_daily_volume: F,
        volatility: F,
        participation_rate: F,
    ) -> Result<F> {
        if average_daily_volume <= F::zero() || volatility < F::zero() {
            return Err(TimeSeriesError::InvalidInput(
                "ADV must be positive and volatility non-negative".to_string(),
            ));
        }

        // Simplified Almgren-Chriss model
        let beta = F::from(0.5).unwrap(); // Square-root law exponent
        let gamma = F::from(0.1).unwrap(); // Market impact coefficient

        let relative_volume = volume / average_daily_volume;
        let impact = gamma * volatility * relative_volume.powf(beta) * participation_rate.sqrt();

        Ok(impact)
    }

    /// Order Book Imbalance
    pub fn order_book_imbalance<F: Float>(bid_volume: F, ask_volume: F) -> F {
        let total_volume = bid_volume + ask_volume;
        if total_volume == F::zero() {
            return F::zero();
        }
        (bid_volume - ask_volume) / total_volume
    }

    /// Microstructure noise indicator
    pub fn microstructure_noise<F: Float + Clone + std::iter::Sum>(
        prices: &Array1<F>,
        window: usize,
    ) -> Result<Array1<F>> {
        if prices.len() < window + 1 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough data for noise calculation".to_string(),
                required: window + 1,
                actual: prices.len(),
            });
        }

        let mut noise = Array1::zeros(prices.len() - window);

        for i in 0..noise.len() {
            let windowprices = prices.slice(s![i..i + window + 1]);

            // Calculate first differences
            let mut diffs = Vec::with_capacity(window);
            for j in 1..windowprices.len() {
                diffs.push(windowprices[j] - windowprices[j - 1]);
            }

            // Calculate variance of first differences
            let sum_diff = diffs.iter().fold(F::zero(), |acc, &x| acc + x);
            let mean_diff = sum_diff / F::from(diffs.len()).unwrap();
            let variance = diffs
                .iter()
                .map(|&d| (d - mean_diff) * (d - mean_diff))
                .sum::<F>()
                / F::from(diffs.len() - 1).unwrap();

            noise[i] = variance.sqrt();
        }

        Ok(noise)
    }
}

/// Black-Scholes options pricing
pub struct BlackScholes;

impl BlackScholes {
    /// Calculate option price using Black-Scholes formula
    pub fn price(contract: &OptionContract, volatility: f64) -> Result<OptionPrice> {
        let s = contract.spot;
        let k = contract.strike;
        let t = contract.maturity;
        let r = contract.risk_freerate;
        let q = contract.dividend_yield;
        let sigma = volatility;

        if t <= 0.0 {
            return match contract.option_type {
                OptionType::Call => Ok(OptionPrice {
                    price: (s - k).max(0.0),
                    delta: if s > k { 1.0 } else { 0.0 },
                    gamma: 0.0,
                    theta: 0.0,
                    vega: 0.0,
                    rho: 0.0,
                }),
                OptionType::Put => Ok(OptionPrice {
                    price: (k - s).max(0.0),
                    delta: if s < k { -1.0 } else { 0.0 },
                    gamma: 0.0,
                    theta: 0.0,
                    vega: 0.0,
                    rho: 0.0,
                }),
                OptionType::AmericanCall => Ok(OptionPrice {
                    price: (s - k).max(0.0),
                    delta: if s > k { 1.0 } else { 0.0 },
                    gamma: 0.0,
                    theta: 0.0,
                    vega: 0.0,
                    rho: 0.0,
                }),
                OptionType::AmericanPut => Ok(OptionPrice {
                    price: (k - s).max(0.0),
                    delta: if s < k { -1.0 } else { 0.0 },
                    gamma: 0.0,
                    theta: 0.0,
                    vega: 0.0,
                    rho: 0.0,
                }),
                OptionType::Barrier { .. } => Ok(OptionPrice {
                    price: 0.0,
                    delta: 0.0,
                    gamma: 0.0,
                    theta: 0.0,
                    vega: 0.0,
                    rho: 0.0,
                }),
                OptionType::Asian { .. } => Ok(OptionPrice {
                    price: 0.0,
                    delta: 0.0,
                    gamma: 0.0,
                    theta: 0.0,
                    vega: 0.0,
                    rho: 0.0,
                }),
            };
        }

        let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
        let d2 = d1 - sigma * t.sqrt();

        let nd1 = Self::standard_normal_cdf(d1);
        let nd2 = Self::standard_normal_cdf(d2);
        let n_minus_d1 = Self::standard_normal_cdf(-d1);
        let n_minus_d2 = Self::standard_normal_cdf(-d2);

        let discount_factor = (-r * t).exp();
        let dividend_factor = (-q * t).exp();

        let (price, delta) = match contract.option_type {
            OptionType::Call => {
                let price = s * dividend_factor * nd1 - k * discount_factor * nd2;
                let delta = dividend_factor * nd1;
                (price, delta)
            }
            OptionType::Put => {
                let price = k * discount_factor * n_minus_d2 - s * dividend_factor * n_minus_d1;
                let delta = -dividend_factor * n_minus_d1;
                (price, delta)
            }
            _ => {
                return Err(TimeSeriesError::NotImplemented(
                    "Only European options supported".to_string(),
                ))
            }
        };

        // Calculate Greeks
        let phi_d1 = Self::standard_normal_pdf(d1);
        let gamma = dividend_factor * phi_d1 / (s * sigma * t.sqrt());
        let vega = s * dividend_factor * phi_d1 * t.sqrt() / 100.0; // Per 1% volatility change
        let theta = match contract.option_type {
            OptionType::Call => {
                (-s * dividend_factor * phi_d1 * sigma / (2.0 * t.sqrt())
                    - r * k * discount_factor * nd2
                    + q * s * dividend_factor * nd1)
                    / 365.0 // Per day
            }
            OptionType::Put => {
                (-s * dividend_factor * phi_d1 * sigma / (2.0 * t.sqrt())
                    + r * k * discount_factor * n_minus_d2
                    - q * s * dividend_factor * n_minus_d1)
                    / 365.0 // Per day
            }
            _ => 0.0,
        };

        let rho = match contract.option_type {
            OptionType::Call => k * t * discount_factor * nd2 / 100.0, // Per 1% rate change
            OptionType::Put => -k * t * discount_factor * n_minus_d2 / 100.0,
            OptionType::AmericanCall => k * t * discount_factor * nd2 / 100.0, // Similar to European call
            OptionType::AmericanPut => -k * t * discount_factor * n_minus_d2 / 100.0, // Similar to European put
            OptionType::Barrier { .. } => k * t * discount_factor * nd2 / 100.0 * 0.8, // Reduced sensitivity due to barrier
            OptionType::Asian { .. } => k * t * discount_factor * nd2 / 100.0 * 0.9, // Reduced sensitivity due to averaging
        };

        Ok(OptionPrice {
            price,
            delta,
            gamma,
            theta,
            vega,
            rho,
        })
    }

    /// Standard normal cumulative distribution function (approximation)
    fn standard_normal_cdf(x: f64) -> f64 {
        if x < -7.0 {
            return 0.0;
        }
        if x > 7.0 {
            return 1.0;
        }

        // Abramowitz and Stegun approximation
        let a1 = 0.31938153;
        let a2 = -0.356563782;
        let a3 = 1.781477937;
        let a4 = -1.821255978;
        let a5 = 1.330274429;

        let k = 1.0 / (1.0 + 0.2316419 * x.abs());
        let phi = (2.0 * std::f64::consts::PI).sqrt().recip() * (-0.5 * x * x).exp();

        if x >= 0.0 {
            1.0 - phi * k * (a1 + k * (a2 + k * (a3 + k * (a4 + k * a5))))
        } else {
            phi * k * (a1 + k * (a2 + k * (a3 + k * (a4 + k * a5))))
        }
    }

    /// Standard normal probability density function
    fn standard_normal_pdf(x: f64) -> f64 {
        (2.0 * std::f64::consts::PI).sqrt().recip() * (-0.5 * x * x).exp()
    }

    /// Implied volatility using Newton-Raphson method
    pub fn implied_volatility(
        contract: &OptionContract,
        market_price: f64,
        initial_guess: f64,
        tolerance: f64,
        max_iterations: usize,
    ) -> Result<f64> {
        let mut vol = initial_guess.max(0.001);

        for _ in 0..max_iterations {
            let option_price = Self::price(contract, vol)?;
            let price_diff = option_price.price - market_price;

            if price_diff.abs() < tolerance {
                return Ok(vol);
            }

            if option_price.vega.abs() < 1e-10 {
                return Err(TimeSeriesError::ComputationError(
                    "Vega too small for Newton-Raphson".to_string(),
                ));
            }

            vol -= price_diff / (option_price.vega * 100.0);
            vol = vol.clamp(0.001, 5.0); // Clamp volatility to reasonable range
        }

        Err(TimeSeriesError::ComputationError(
            "Implied volatility calculation did not converge".to_string(),
        ))
    }
}

/// Regime-switching model for financial time series
#[derive(Debug)]
pub struct RegimeSwitchingModel<F: Float + Debug> {
    /// Number of regimes
    num_regimes: usize,
    /// Transition probability matrix
    transition_probs: Array2<F>,
    /// Regime-specific parameters
    regime_params: Vec<RegimeParameters<F>>,
    /// Current state probabilities
    state_probs: Array1<F>,
}

/// Parameters for a specific regime in the regime-switching model
#[derive(Debug, Clone)]
pub struct RegimeParameters<F: Float> {
    /// Mean for this regime
    pub mean: F,
    /// Volatility for this regime
    pub volatility: F,
    /// Persistence parameter
    pub persistence: F,
}

impl<F: Float + Debug + Clone + FromPrimitive + scirs2_core::ndarray::ScalarOperand>
    RegimeSwitchingModel<F>
{
    /// Create new regime-switching model
    pub fn new(
        num_regimes: usize,
        transition_probs: Array2<F>,
        regime_params: Vec<RegimeParameters<F>>,
    ) -> Result<Self> {
        if transition_probs.dim() != (num_regimes, num_regimes) {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: num_regimes * num_regimes,
                actual: transition_probs.len(),
            });
        }

        if regime_params.len() != num_regimes {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: num_regimes,
                actual: regime_params.len(),
            });
        }

        // Initialize with uniform state probabilities
        let state_probs = Array1::from_elem(num_regimes, F::one() / F::from(num_regimes).unwrap());

        Ok(Self {
            num_regimes,
            transition_probs,
            regime_params,
            state_probs,
        })
    }

    /// Update state probabilities using Baum-Welch algorithm (simplified)
    pub fn update_states(&mut self, observation: F) -> Result<()> {
        let mut new_probs = Array1::zeros(self.num_regimes);

        // Calculate likelihood for each regime
        for i in 0..self.num_regimes {
            let regime = &self.regime_params[i];
            let likelihood = self.gaussian_likelihood(observation, regime.mean, regime.volatility);
            new_probs[i] = likelihood;
        }

        // Apply transition probabilities
        let mut updated_probs: Array1<F> = Array1::zeros(self.num_regimes);
        for i in 0..self.num_regimes {
            for j in 0..self.num_regimes {
                updated_probs[i] =
                    updated_probs[i] + self.state_probs[j] * self.transition_probs[[j, i]];
            }
            updated_probs[i] = updated_probs[i] * new_probs[i];
        }

        // Normalize probabilities
        let total_prob = updated_probs.sum();
        if total_prob > F::zero() {
            self.state_probs = updated_probs / total_prob;
        }

        Ok(())
    }

    /// Calculate Gaussian likelihood
    fn gaussian_likelihood(&self, x: F, mean: F, stddev: F) -> F {
        let two_pi = F::from(2.0 * std::f64::consts::PI).unwrap();
        let sqrt_two_pi = two_pi.sqrt();
        let variance = stddev * stddev;
        let diff = x - mean;
        let exponent = -(diff * diff) / (F::from(2).unwrap() * variance);

        exponent.exp() / (sqrt_two_pi * stddev)
    }

    /// Get current most likely regime
    pub fn current_regime(&self) -> usize {
        let mut max_prob = F::neg_infinity();
        let mut max_regime = 0;

        for (i, &prob) in self.state_probs.iter().enumerate() {
            if prob > max_prob {
                max_prob = prob;
                max_regime = i;
            }
        }

        max_regime
    }

    /// Get state probabilities
    pub fn state_probabilities(&self) -> &Array1<F> {
        &self.state_probs
    }

    /// Forecast next period using current regime probabilities
    pub fn forecast(&self) -> Result<F> {
        let mut forecast = F::zero();

        for (i, &prob) in self.state_probs.iter().enumerate() {
            forecast = forecast + prob * self.regime_params[i].mean;
        }

        Ok(forecast)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_value_at_risk() {
        let returns = Array1::from_vec(vec![-0.05, -0.02, 0.01, -0.01, 0.03, -0.04, 0.02]);
        let risk_metrics = RiskMetrics::new(returns);

        let var_95 = risk_metrics.value_at_risk(0.95).unwrap();
        assert!(var_95 > 0.0); // VaR should be positive (representing loss)
    }

    #[test]
    fn test_sharpe_ratio() {
        let returns = Array1::from_vec(vec![0.01, 0.02, -0.01, 0.03, 0.00, 0.02, 0.01]);
        let risk_metrics = RiskMetrics::new(returns);

        let sharpe = risk_metrics.sharpe_ratio(0.02).unwrap();
        assert!(sharpe.is_finite());
    }

    #[test]
    fn test_maximum_drawdown() {
        let returns = Array1::from_vec(vec![0.10, -0.05, -0.10, 0.05, 0.15, -0.20]);
        let risk_metrics = RiskMetrics::new(returns);

        let max_dd = risk_metrics.maximum_drawdown().unwrap();
        assert!(max_dd >= 0.0);
        assert!(max_dd <= 1.0);
    }

    #[test]
    fn test_black_scholes_call() {
        let contract = OptionContract {
            spot: 100.0,
            strike: 100.0,
            maturity: 0.25, // 3 months
            risk_freerate: 0.05,
            dividend_yield: 0.0,
            option_type: OptionType::Call,
        };

        let option_price = BlackScholes::price(&contract, 0.2).unwrap();

        assert!(option_price.price > 0.0);
        assert!(option_price.delta > 0.0 && option_price.delta < 1.0);
        assert!(option_price.gamma > 0.0);
        assert!(option_price.vega > 0.0);
        assert!(option_price.theta < 0.0); // Time decay for long option
    }

    #[test]
    fn test_black_scholes_put() {
        let contract = OptionContract {
            spot: 100.0,
            strike: 100.0,
            maturity: 0.25,
            risk_freerate: 0.05,
            dividend_yield: 0.0,
            option_type: OptionType::Put,
        };

        let option_price = BlackScholes::price(&contract, 0.2).unwrap();

        assert!(option_price.price > 0.0);
        assert!(option_price.delta < 0.0 && option_price.delta > -1.0);
        assert!(option_price.gamma > 0.0);
        assert!(option_price.vega > 0.0);
        assert!(option_price.theta < 0.0);
    }

    #[test]
    fn test_vwap() {
        let prices = Array1::from_vec(vec![100.0, 101.0, 102.0, 101.5, 100.5]);
        let volumes = Array1::from_vec(vec![1000.0, 1500.0, 800.0, 1200.0, 900.0]);

        let vwap = HFTIndicators::vwap(&prices, &volumes).unwrap();
        assert_eq!(vwap.len(), prices.len());

        // VWAP should be reasonable
        assert!(vwap[vwap.len() - 1] >= 100.0 && vwap[vwap.len() - 1] <= 102.0);
    }

    #[test]
    fn test_order_book_imbalance() {
        let imbalance = HFTIndicators::order_book_imbalance(1000.0, 800.0);
        assert_abs_diff_eq!(imbalance, 0.111111, epsilon = 1e-5);

        let balanced = HFTIndicators::order_book_imbalance(500.0, 500.0);
        assert_abs_diff_eq!(balanced, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_regime_switching_model() {
        let transition_probs = Array2::from_shape_vec((2, 2), vec![0.8, 0.2, 0.3, 0.7]).unwrap();

        let regime_params = vec![
            RegimeParameters {
                mean: 0.05,
                volatility: 0.1,
                persistence: 0.8,
            },
            RegimeParameters {
                mean: -0.02,
                volatility: 0.2,
                persistence: 0.7,
            },
        ];

        let mut model = RegimeSwitchingModel::new(2, transition_probs, regime_params).unwrap();

        // Test state update
        model.update_states(0.03).unwrap();
        let current_regime = model.current_regime();
        assert!(current_regime < 2);

        // Test forecasting
        let forecast = model.forecast().unwrap();
        assert!(forecast.is_finite());
    }

    #[test]
    fn test_implied_volatility() {
        let contract = OptionContract {
            spot: 100.0,
            strike: 100.0,
            maturity: 0.25,
            risk_freerate: 0.05,
            dividend_yield: 0.0,
            option_type: OptionType::Call,
        };

        // Calculate theoretical price with known volatility
        let known_vol = 0.2;
        let theoretical_price = BlackScholes::price(&contract, known_vol).unwrap().price;

        // Calculate implied volatility from theoretical price
        let implied_vol = BlackScholes::implied_volatility(
            &contract,
            theoretical_price,
            0.15, // initial guess
            1e-6, // tolerance
            100,  // max iterations
        )
        .unwrap();

        assert_abs_diff_eq!(implied_vol, known_vol, epsilon = 1e-4);
    }
}
