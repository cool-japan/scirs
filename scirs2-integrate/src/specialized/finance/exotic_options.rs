//! Exotic option pricing facade
//!
//! This module unifies
//! [`derivatives::exotic`](crate::specialized::finance::derivatives::exotic)'s
//! [`BarrierOption`], [`AsianOption`], [`LookbackOption`], and
//! [`DigitalOption`] behind a single enum dispatcher ([`ExoticOptionType`])
//! and pricer ([`ExoticOptionPricer`]) that returns one common result shape
//! ([`PricingResult`]) regardless of which exotic payoff was priced.
//!
//! `RainbowPayoffType` (multi-asset correlated-path pricing) is intentionally
//! NOT implemented here — see `TODO.md` "Proposed follow-ups" for why.

use crate::error::{IntegrateError, IntegrateResult};
use crate::specialized::finance::derivatives::exotic::{
    AsianOption, BarrierOption, DigitalOption, LookbackOption,
};

// ============================================================
// ExoticOptionType
// ============================================================

/// Discriminated union over the concrete exotic option contracts, used to
/// dispatch pricing through a single [`ExoticOptionPricer`] entry point.
#[derive(Debug, Clone)]
pub enum ExoticOptionType {
    /// Knock-in/knock-out barrier option.
    Barrier(BarrierOption),
    /// Average-rate (Asian) option.
    Asian(AsianOption),
    /// Fixed/floating-strike lookback option.
    Lookback(LookbackOption),
    /// Cash-or-nothing / asset-or-nothing digital option.
    Digital(DigitalOption),
}

impl ExoticOptionType {
    /// Human-readable label for the wrapped variant.
    pub fn label(&self) -> &'static str {
        match self {
            ExoticOptionType::Barrier(_) => "Barrier",
            ExoticOptionType::Asian(_) => "Asian",
            ExoticOptionType::Lookback(_) => "Lookback",
            ExoticOptionType::Digital(_) => "Digital",
        }
    }
}

// ============================================================
// PricingResult
// ============================================================

/// Method used to arrive at a [`PricingResult`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PricingMethod {
    /// Monte Carlo simulation with the given path/step counts.
    MonteCarlo {
        /// Number of simulated paths.
        n_paths: usize,
        /// Number of time steps per path (unused for options whose
        /// underlying pricer only takes a path count, e.g. Asian).
        n_steps: usize,
    },
    /// An analytical closed-form formula (no simulation).
    ClosedForm,
}

/// Common result shape returned by [`ExoticOptionPricer`], regardless of
/// which exotic option variant was priced.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PricingResult {
    /// The computed price.
    pub price: f64,
    /// How the price was computed.
    pub method: PricingMethod,
    /// Label of the exotic option variant that was priced.
    pub option_label: &'static str,
}

// ============================================================
// ExoticOptionPricer
// ============================================================

/// Unifying facade over the exotic option contract types.
///
/// Dispatches to each contract's own pricing routine and normalizes the
/// result into a common [`PricingResult`] shape.
#[derive(Debug, Clone)]
pub struct ExoticOptionPricer {
    /// The wrapped exotic option contract.
    pub option: ExoticOptionType,
}

impl ExoticOptionPricer {
    /// Wrap an exotic option contract for unified pricing.
    pub fn new(option: ExoticOptionType) -> Self {
        Self { option }
    }

    /// Price the wrapped exotic option via Monte Carlo simulation.
    ///
    /// `n_steps` is ignored for [`ExoticOptionType::Asian`] (its underlying
    /// `price_monte_carlo` only takes a path count). [`ExoticOptionType::Digital`]
    /// is always priced in closed form (no simulation is needed), and the
    /// reported [`PricingMethod`] reflects that regardless of the requested
    /// `n_paths`/`n_steps`.
    pub fn price(&self, n_paths: usize, n_steps: usize) -> IntegrateResult<PricingResult> {
        match &self.option {
            ExoticOptionType::Barrier(opt) => Ok(PricingResult {
                price: opt.price_monte_carlo(n_paths, n_steps)?,
                method: PricingMethod::MonteCarlo { n_paths, n_steps },
                option_label: self.option.label(),
            }),
            ExoticOptionType::Asian(opt) => Ok(PricingResult {
                price: opt.price_monte_carlo(n_paths)?,
                method: PricingMethod::MonteCarlo { n_paths, n_steps },
                option_label: self.option.label(),
            }),
            ExoticOptionType::Lookback(opt) => Ok(PricingResult {
                price: opt.price_monte_carlo(n_paths, n_steps)?,
                method: PricingMethod::MonteCarlo { n_paths, n_steps },
                option_label: self.option.label(),
            }),
            ExoticOptionType::Digital(opt) => Ok(PricingResult {
                price: opt.price(),
                method: PricingMethod::ClosedForm,
                option_label: self.option.label(),
            }),
        }
    }

    /// Price using a closed-form formula where one is available (geometric
    /// Asian, Digital). Returns [`IntegrateError::NotImplementedError`] for
    /// variants that have no closed-form implementation (Barrier, Lookback,
    /// and arithmetic-averaging Asian — see
    /// [`AsianOption::price_geometric_closed_form`] for the geometric-only
    /// restriction).
    pub fn price_closed_form(&self) -> IntegrateResult<PricingResult> {
        match &self.option {
            ExoticOptionType::Asian(opt) => Ok(PricingResult {
                price: opt.price_geometric_closed_form()?,
                method: PricingMethod::ClosedForm,
                option_label: self.option.label(),
            }),
            ExoticOptionType::Digital(opt) => Ok(PricingResult {
                price: opt.price(),
                method: PricingMethod::ClosedForm,
                option_label: self.option.label(),
            }),
            ExoticOptionType::Barrier(_) | ExoticOptionType::Lookback(_) => {
                Err(IntegrateError::NotImplementedError(format!(
                    "Closed-form pricing is not available for {} options",
                    self.option.label()
                )))
            }
        }
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::specialized::finance::derivatives::exotic::{
        AveragingMethod, BarrierType, DigitalType,
    };
    use crate::specialized::finance::types::OptionType;
    use approx::assert_relative_eq;

    #[test]
    fn test_digital_facade_matches_direct_closed_form() {
        let digital = DigitalOption::new(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            OptionType::Call,
            DigitalType::CashOrNothing { cash_amount: 10.0 },
        )
        .expect("digital construction failed");

        let direct_price = digital.price();

        let pricer = ExoticOptionPricer::new(ExoticOptionType::Digital(digital));
        let facade_result = pricer.price(0, 0).expect("facade pricing failed");

        assert_relative_eq!(facade_result.price, direct_price, epsilon = 1e-14);
        assert_eq!(facade_result.method, PricingMethod::ClosedForm);
        assert_eq!(facade_result.option_label, "Digital");

        let cf_result = pricer.price_closed_form().expect("closed form failed");
        assert_relative_eq!(cf_result.price, direct_price, epsilon = 1e-14);
    }

    #[test]
    fn test_asian_geometric_facade_matches_direct_closed_form() {
        let asian = AsianOption::new(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            OptionType::Call,
            AveragingMethod::Geometric,
            50,
        )
        .expect("asian construction failed");

        let direct_price = asian
            .price_geometric_closed_form()
            .expect("direct closed form failed");

        let pricer = ExoticOptionPricer::new(ExoticOptionType::Asian(asian));
        let facade_result = pricer
            .price_closed_form()
            .expect("facade closed form failed");

        assert_relative_eq!(facade_result.price, direct_price, epsilon = 1e-14);
        assert_eq!(facade_result.method, PricingMethod::ClosedForm);
        assert_eq!(facade_result.option_label, "Asian");
    }

    #[test]
    fn test_barrier_facade_has_no_closed_form_but_prices_via_monte_carlo() {
        let barrier = BarrierOption::new(
            100.0,
            100.0,
            120.0,
            0.0,
            0.05,
            0.0,
            0.2,
            1.0,
            OptionType::Call,
            BarrierType::UpAndOut,
        )
        .expect("barrier construction failed");

        let pricer = ExoticOptionPricer::new(ExoticOptionType::Barrier(barrier));
        assert!(pricer.price_closed_form().is_err());

        let result = pricer.price(2000, 100).expect("mc pricing failed");
        assert!(result.price >= 0.0);
        assert_eq!(
            result.method,
            PricingMethod::MonteCarlo {
                n_paths: 2000,
                n_steps: 100
            }
        );
        assert_eq!(result.option_label, "Barrier");
    }
}
