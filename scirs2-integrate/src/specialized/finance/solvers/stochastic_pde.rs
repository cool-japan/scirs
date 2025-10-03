//! Main stochastic PDE solver for financial derivatives

use crate::error::IntegrateResult;
use crate::specialized::finance::models::{JumpProcess, StochasticProcess, VolatilityModel};
use crate::specialized::finance::types::{FinanceMethod, FinancialOption, OptionType};
use scirs2_core::ndarray::Array2;

/// Solver for stochastic PDEs in finance
pub struct StochasticPDESolver {
    /// Grid points in asset dimension
    pub n_asset: usize,
    /// Grid points in time dimension
    pub n_time: usize,
    /// Grid points in volatility dimension (for stochastic vol)
    pub n_vol: Option<usize>,
    /// Volatility model
    pub volatility_model: VolatilityModel,
    /// Jump process (optional)
    pub jump_process: Option<JumpProcess>,
    /// Underlying stochastic process
    pub stochastic_process: Option<StochasticProcess>,
    /// Solver method
    pub method: FinanceMethod,
}

impl StochasticPDESolver {
    /// Create a new stochastic PDE solver
    pub fn new(
        n_asset: usize,
        n_time: usize,
        volatility_model: VolatilityModel,
        method: FinanceMethod,
    ) -> Self {
        let n_vol = match &volatility_model {
            VolatilityModel::Heston { .. }
            | VolatilityModel::SABR { .. }
            | VolatilityModel::Bates { .. }
            | VolatilityModel::HullWhite { .. }
            | VolatilityModel::ThreeHalves { .. } => Some(50),
            _ => None,
        };

        Self {
            n_asset,
            n_time,
            n_vol,
            volatility_model,
            jump_process: None,
            stochastic_process: None,
            method,
        }
    }

    /// Add jump process
    pub fn with_jumps(mut self, jump_process: JumpProcess) -> Self {
        self.jump_process = Some(jump_process);
        self
    }

    /// Set underlying stochastic process
    pub fn with_stochastic_process(mut self, process: StochasticProcess) -> Self {
        self.stochastic_process = Some(process);
        self
    }

    /// Price option using specified method
    pub fn price_option(&self, option: &FinancialOption) -> IntegrateResult<f64> {
        match self.method {
            FinanceMethod::FiniteDifference => self.price_finite_difference(option),
            FinanceMethod::MonteCarlo {
                n_paths,
                antithetic,
            } => self.price_monte_carlo(option, n_paths, antithetic),
            FinanceMethod::FourierTransform => self.price_fourier_transform(option),
            FinanceMethod::Tree { n_steps } => self.price_tree(option, n_steps),
        }
    }

    /// Calculate option payoff
    pub(crate) fn payoff(&self, option: &FinancialOption, spot: f64) -> f64 {
        match option.option_type {
            OptionType::Call => (spot - option.strike).max(0.0),
            OptionType::Put => (option.strike - spot).max(0.0),
        }
    }

    // Method signatures - implementations will be in separate modules
    fn price_finite_difference(&self, option: &FinancialOption) -> IntegrateResult<f64> {
        use crate::specialized::finance::pricing::finite_difference;
        finite_difference::price_finite_difference(self, option)
    }

    fn price_monte_carlo(
        &self,
        option: &FinancialOption,
        n_paths: usize,
        antithetic: bool,
    ) -> IntegrateResult<f64> {
        use crate::specialized::finance::pricing::monte_carlo;
        monte_carlo::price_monte_carlo(self, option, n_paths, antithetic)
    }

    fn price_fourier_transform(&self, option: &FinancialOption) -> IntegrateResult<f64> {
        use crate::specialized::finance::pricing::fourier;
        fourier::price_fourier_transform(self, option)
    }

    fn price_tree(&self, option: &FinancialOption, n_steps: usize) -> IntegrateResult<f64> {
        use crate::specialized::finance::pricing::tree;
        tree::price_tree(self, option, n_steps)
    }
}
