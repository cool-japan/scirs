//! Advanced Monte Carlo pricing engine facade
//!
//! This module wraps [`crate::specialized::finance::monte_carlo::MonteCarloEngine`]
//! and packages its variance reduction techniques — antithetic variates,
//! control variates, and Sobol low-discrepancy sequences — into a single
//! configurable [`VarianceReductionSuite`]. [`MonteCarloResult`] is
//! thin-wrapped as [`OptionPricingResult`], which additionally records which
//! techniques were applied, for API-naming consistency with the
//! originally-envisioned (but never implemented) `advanced_monte_carlo_engine`
//! module referenced from `specialized::mod`.
//!
//! # Sobol variance reduction — scope
//!
//! The existing [`sobol_sequence`] helper supports at most 10 dimensions.
//! A full multi-step GBM path (as used by the Asian/Barrier/Lookback/American
//! pricers) needs one dimension per time step, which routinely exceeds that
//! cap (e.g. 252 daily steps). Rather than inventing a new high-dimensional
//! low-discrepancy construction (Brownian bridge, dimension reduction, …),
//! this facade applies Sobol only where it is mathematically sound *and*
//! bounded by the existing 10-dimension implementation: European-style
//! payoffs under GBM depend only on the terminal price `S_T`, which has a
//! closed-form single-draw representation
//! `S_T = S0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)`. A 1-dimensional
//! Sobol sequence, inverted through the Beasley-Springer-Moro approximation
//! already used by [`HistoricalVaR`](crate::specialized::finance::risk::var::HistoricalVaR)-adjacent
//! parametric VaR, supplies `Z` deterministically. Other pricers in this
//! facade (Asian, Barrier, Lookback) ignore the Sobol flag and always use
//! the wrapped engine's pseudo-random path generator, since they are
//! path-dependent and thus outside this bounded scope.
//!
//! `QuantumInspiredRNG` is intentionally NOT implemented here — see
//! `TODO.md` "Proposed follow-ups" for why.

use crate::error::{IntegrateError, IntegrateResult};
use crate::specialized::finance::monte_carlo::{
    build_result, mc_asian_option, mc_barrier_option, mc_european_option, mc_greeks,
    mc_lookback_option, sobol_sequence, AsianAveraging, BarrierType, MonteCarloEngine,
    MonteCarloResult, OptionGreeks, OptionType,
};
use crate::specialized::finance::risk::var::inverse_normal_cdf;

// ============================================================
// VarianceReductionSuite
// ============================================================

/// Configuration selecting which variance-reduction techniques an
/// [`AdvancedMonteCarloEngine`] should apply.
///
/// This packages the techniques already implemented by
/// [`MonteCarloEngine`] (antithetic variates, control variates) alongside
/// the standalone [`sobol_sequence`] low-discrepancy generator into one
/// builder-style configuration value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VarianceReductionSuite {
    /// Use antithetic variates (mirrored shocks) to halve simulation noise.
    pub antithetic: bool,
    /// Use the Black-Scholes / geometric-Asian analytical price as a control
    /// variate correction.
    pub control_variates: bool,
    /// Use a Sobol low-discrepancy sequence in place of pseudo-random
    /// draws (European terminal-distribution pricing only — see module docs).
    pub sobol: bool,
}

impl VarianceReductionSuite {
    /// No variance reduction (plain pseudo-random Monte Carlo).
    pub fn none() -> Self {
        Self::default()
    }

    /// Enable every supported technique.
    pub fn all() -> Self {
        Self {
            antithetic: true,
            control_variates: true,
            sobol: true,
        }
    }

    /// Enable antithetic variates.
    pub fn with_antithetic(mut self) -> Self {
        self.antithetic = true;
        self
    }

    /// Enable control variates.
    pub fn with_control_variates(mut self) -> Self {
        self.control_variates = true;
        self
    }

    /// Enable Sobol low-discrepancy sampling (European terminal draw only).
    pub fn with_sobol(mut self) -> Self {
        self.sobol = true;
        self
    }

    /// True if any technique is enabled.
    pub fn any(&self) -> bool {
        self.antithetic || self.control_variates || self.sobol
    }
}

// ============================================================
// OptionPricingResult
// ============================================================

/// Option pricing result returned by [`AdvancedMonteCarloEngine`].
///
/// Thin wrapper around [`MonteCarloResult`] that additionally records which
/// [`VarianceReductionSuite`] was used to produce it.
#[derive(Debug, Clone)]
pub struct OptionPricingResult {
    /// Option price estimate.
    pub price: f64,
    /// Standard error of the Monte Carlo estimate.
    pub std_error: f64,
    /// 95% confidence interval (lower, upper).
    pub confidence_interval: (f64, f64),
    /// Number of paths used.
    pub n_paths: usize,
    /// Variance-reduction techniques applied to produce this result.
    pub variance_reduction: VarianceReductionSuite,
}

impl OptionPricingResult {
    fn from_result(result: MonteCarloResult, variance_reduction: VarianceReductionSuite) -> Self {
        Self {
            price: result.price,
            std_error: result.std_error,
            confidence_interval: result.confidence_interval,
            n_paths: result.n_paths,
            variance_reduction,
        }
    }
}

// ============================================================
// AdvancedMonteCarloEngine
// ============================================================

/// Advanced Monte Carlo pricing engine facade.
///
/// Wraps a [`MonteCarloEngine`], configured via a [`VarianceReductionSuite`],
/// and exposes the same option-pricing surface
/// (European, Asian, Barrier, Lookback, Greeks) with results reported as
/// [`OptionPricingResult`].
#[derive(Debug, Clone)]
pub struct AdvancedMonteCarloEngine {
    engine: MonteCarloEngine,
    variance_reduction: VarianceReductionSuite,
}

impl AdvancedMonteCarloEngine {
    /// Create a new engine with no variance reduction enabled.
    pub fn new(n_paths: usize, n_steps: usize, seed: u64) -> Self {
        Self {
            engine: MonteCarloEngine::new(n_paths, n_steps, seed),
            variance_reduction: VarianceReductionSuite::none(),
        }
    }

    /// Configure which variance-reduction techniques to apply.
    ///
    /// Antithetic and control-variate flags are forwarded directly to the
    /// wrapped [`MonteCarloEngine`]; the Sobol flag is handled by this
    /// facade (see module docs for scope).
    pub fn with_variance_reduction(mut self, suite: VarianceReductionSuite) -> Self {
        self.engine.antithetic = suite.antithetic;
        self.engine.control_variates = suite.control_variates;
        self.variance_reduction = suite;
        self
    }

    /// Read-only access to the wrapped engine (paths/steps/seed/flags).
    pub fn engine(&self) -> &MonteCarloEngine {
        &self.engine
    }

    /// Currently configured variance-reduction suite.
    pub fn variance_reduction(&self) -> VarianceReductionSuite {
        self.variance_reduction
    }

    /// Price a European call or put.
    ///
    /// When [`VarianceReductionSuite::sobol`] is enabled, uses a
    /// 1-dimensional Sobol sequence for the terminal draw (see module docs);
    /// otherwise delegates directly to
    /// [`mc_european_option`].
    pub fn price_european(
        &self,
        s0: f64,
        k: f64,
        r: f64,
        sigma: f64,
        t: f64,
        option_type: OptionType,
    ) -> IntegrateResult<OptionPricingResult> {
        let result = if self.variance_reduction.sobol {
            self.price_european_sobol(s0, k, r, sigma, t, option_type)?
        } else {
            mc_european_option(s0, k, r, sigma, t, option_type, &self.engine)?
        };
        Ok(OptionPricingResult::from_result(
            result,
            self.variance_reduction,
        ))
    }

    /// European terminal-distribution pricer driven by a Sobol
    /// low-discrepancy sequence instead of pseudo-random draws.
    fn price_european_sobol(
        &self,
        s0: f64,
        k: f64,
        r: f64,
        sigma: f64,
        t: f64,
        option_type: OptionType,
    ) -> IntegrateResult<MonteCarloResult> {
        if s0 <= 0.0 {
            return Err(IntegrateError::ValueError(
                "Initial stock price must be positive".to_string(),
            ));
        }
        if sigma < 0.0 {
            return Err(IntegrateError::ValueError(
                "Volatility must be non-negative".to_string(),
            ));
        }
        if t <= 0.0 {
            return Err(IntegrateError::ValueError(
                "Time to maturity must be positive".to_string(),
            ));
        }
        if self.engine.n_paths == 0 {
            return Err(IntegrateError::ValueError(
                "n_paths must be positive".to_string(),
            ));
        }

        let drift = (r - 0.5 * sigma * sigma) * t;
        let vol_sqrt_t = sigma * t.sqrt();
        let is_call = option_type == OptionType::Call;

        // 1-D Sobol sequence: one quasi-random uniform per path, inverted to
        // a standard normal via the Beasley-Springer-Moro approximation.
        let sobol = sobol_sequence(self.engine.n_paths, 1, 0);
        let payoffs: Vec<f64> = sobol
            .column(0)
            .iter()
            .map(|&u| {
                // Keep away from the 0/1 boundary where the inverse CDF diverges.
                let u_clamped = u.clamp(1e-10, 1.0 - 1e-10);
                let z = inverse_normal_cdf(u_clamped);
                let s_t = s0 * (drift + vol_sqrt_t * z).exp();
                if is_call {
                    (s_t - k).max(0.0)
                } else {
                    (k - s_t).max(0.0)
                }
            })
            .collect();

        Ok(build_result(&payoffs, self.engine.n_paths, r, t))
    }

    /// Price an Asian (average-rate) option. Delegates to
    /// [`mc_asian_option`]; Sobol is not
    /// applicable (path-dependent payoff — see module docs).
    pub fn price_asian(
        &self,
        s0: f64,
        k: f64,
        r: f64,
        sigma: f64,
        t: f64,
        option_type: OptionType,
        averaging: AsianAveraging,
    ) -> IntegrateResult<OptionPricingResult> {
        let result = mc_asian_option(s0, k, r, sigma, t, option_type, averaging, &self.engine)?;
        Ok(OptionPricingResult::from_result(
            result,
            self.variance_reduction,
        ))
    }

    /// Price a barrier option. Delegates to
    /// [`mc_barrier_option`]; Sobol is not
    /// applicable (path-dependent payoff — see module docs).
    #[allow(clippy::too_many_arguments)]
    pub fn price_barrier(
        &self,
        s0: f64,
        k: f64,
        r: f64,
        sigma: f64,
        t: f64,
        barrier: f64,
        option_type: OptionType,
        barrier_type: BarrierType,
    ) -> IntegrateResult<OptionPricingResult> {
        let result = mc_barrier_option(
            s0,
            k,
            r,
            sigma,
            t,
            barrier,
            option_type,
            barrier_type,
            &self.engine,
        )?;
        Ok(OptionPricingResult::from_result(
            result,
            self.variance_reduction,
        ))
    }

    /// Price a fixed-strike lookback option. Delegates to
    /// [`mc_lookback_option`]; Sobol is
    /// not applicable (path-dependent payoff — see module docs).
    pub fn price_lookback(
        &self,
        s0: f64,
        k: f64,
        r: f64,
        sigma: f64,
        t: f64,
        option_type: OptionType,
    ) -> IntegrateResult<OptionPricingResult> {
        let result = mc_lookback_option(s0, k, r, sigma, t, option_type, &self.engine)?;
        Ok(OptionPricingResult::from_result(
            result,
            self.variance_reduction,
        ))
    }

    /// Compute European option Greeks via finite differences. Delegates to
    /// [`mc_greeks`].
    pub fn greeks(
        &self,
        s0: f64,
        k: f64,
        r: f64,
        sigma: f64,
        t: f64,
        option_type: OptionType,
    ) -> IntegrateResult<OptionGreeks> {
        mc_greeks(s0, k, r, sigma, t, option_type, &self.engine)
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Standard Black-Scholes call price, replicated locally for testing
    /// (mirrors the private helper used throughout
    /// `finance::monte_carlo`'s own test module).
    fn bs_call_reference(s0: f64, k: f64, r: f64, sigma: f64, t: f64) -> f64 {
        let d1 = ((s0 / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
        let d2 = d1 - sigma * t.sqrt();
        let n = |x: f64| 0.5 * (1.0 + libm::erf(x / std::f64::consts::SQRT_2));
        s0 * n(d1) - k * (-r * t).exp() * n(d2)
    }

    #[test]
    fn test_advanced_engine_matches_direct_mc_call() {
        let advanced = AdvancedMonteCarloEngine::new(20_000, 252, 42).with_variance_reduction(
            VarianceReductionSuite::none()
                .with_antithetic()
                .with_control_variates(),
        );
        let facade_result = advanced
            .price_european(100.0, 100.0, 0.05, 0.2, 1.0, OptionType::Call)
            .expect("facade pricing failed");

        let direct_engine = MonteCarloEngine::new(20_000, 252, 42)
            .with_antithetic()
            .with_control_variates();
        let direct_result = mc_european_option(
            100.0,
            100.0,
            0.05,
            0.2,
            1.0,
            OptionType::Call,
            &direct_engine,
        )
        .expect("direct pricing failed");

        // Same engine settings + same seed => the facade must reproduce the
        // direct MonteCarloEngine call bit-for-bit.
        assert_relative_eq!(facade_result.price, direct_result.price, epsilon = 1e-12);
        assert_relative_eq!(
            facade_result.std_error,
            direct_result.std_error,
            epsilon = 1e-12
        );
        assert_eq!(facade_result.n_paths, direct_result.n_paths);
        assert!(facade_result.variance_reduction.antithetic);
        assert!(facade_result.variance_reduction.control_variates);
        assert!(!facade_result.variance_reduction.sobol);
    }

    #[test]
    fn test_advanced_engine_sobol_converges_to_black_scholes() {
        // The Sobol path uses a fundamentally different (quasi-random,
        // terminal-only) sampling scheme, so bit-for-bit equality with the
        // pseudo-random engine isn't meaningful. Instead confirm it converges
        // to the closed-form Black-Scholes reference, consistent with what
        // the underlying (pseudo-random) MonteCarloEngine is already tested
        // to converge to elsewhere in this crate.
        let advanced = AdvancedMonteCarloEngine::new(20_000, 50, 7)
            .with_variance_reduction(VarianceReductionSuite::none().with_sobol());
        let result = advanced
            .price_european(100.0, 100.0, 0.05, 0.2, 1.0, OptionType::Call)
            .expect("sobol pricing failed");

        let bs_price = bs_call_reference(100.0, 100.0, 0.05, 0.2, 1.0);
        assert!(
            (result.price - bs_price).abs() < 0.1 * bs_price,
            "Sobol MC price {} too far from BS reference {}",
            result.price,
            bs_price
        );
        assert!(result.variance_reduction.sobol);
    }

    #[test]
    fn test_advanced_engine_asian_and_greeks_delegate_correctly() {
        let advanced = AdvancedMonteCarloEngine::new(5_000, 100, 11);

        let asian_facade = advanced
            .price_asian(
                100.0,
                100.0,
                0.05,
                0.2,
                1.0,
                OptionType::Call,
                AsianAveraging::Arithmetic,
            )
            .expect("facade asian failed");
        let asian_direct = mc_asian_option(
            100.0,
            100.0,
            0.05,
            0.2,
            1.0,
            OptionType::Call,
            AsianAveraging::Arithmetic,
            advanced.engine(),
        )
        .expect("direct asian failed");
        assert_relative_eq!(asian_facade.price, asian_direct.price, epsilon = 1e-12);

        let greeks_facade = advanced
            .greeks(100.0, 100.0, 0.05, 0.2, 1.0, OptionType::Call)
            .expect("facade greeks failed");
        let greeks_direct = mc_greeks(
            100.0,
            100.0,
            0.05,
            0.2,
            1.0,
            OptionType::Call,
            advanced.engine(),
        )
        .expect("direct greeks failed");
        assert_relative_eq!(greeks_facade.delta, greeks_direct.delta, epsilon = 1e-12);
    }
}
