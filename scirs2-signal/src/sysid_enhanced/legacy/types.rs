//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions::enhanced_system_identification;
use crate::error::{SignalError, SignalResult};
use crate::lti::{StateSpace, TransferFunction};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::prelude::*;

/// Adaptive identification with time-varying parameters
pub struct AdaptiveIdentifier {
    current_model: Option<SystemModel>,
    parameter_history: Vec<Array1<f64>>,
    forgetting_factor: f64,
    adaptation_threshold: f64,
    config: EnhancedSysIdConfig,
}
impl AdaptiveIdentifier {
    pub fn new(config: EnhancedSysIdConfig) -> Self {
        Self {
            current_model: None,
            parameter_history: Vec::new(),
            forgetting_factor: config.forgetting_factor,
            adaptation_threshold: 0.1,
            config,
        }
    }
    pub fn update_model(
        &mut self,
        input: &Array1<f64>,
        output: &Array1<f64>,
    ) -> SignalResult<bool> {
        let result = enhanced_system_identification(input, output, &self.config)?;
        let should_adapt = if let Some(ref current_params) = self.parameter_history.last() {
            if result.parameters.values.len() != current_params.len() {
                // Order/structure selection picked a different parameter
                // count than the previous fit (e.g. the model order changed
                // between windows) — element-wise diffing two different-
                // length vectors would panic (ndarray shape mismatch), and a
                // structural change is itself significant enough to adapt.
                true
            } else {
                let diff = &result.parameters.values - *current_params;
                let diff_norm = diff.iter().map(|&x| x * x).sum::<f64>().sqrt();
                let curr_norm = current_params.iter().map(|&x| x * x).sum::<f64>().sqrt();
                let param_change = diff_norm / curr_norm.max(1e-15);
                param_change > self.adaptation_threshold
            }
        } else {
            true
        };
        if should_adapt {
            self.current_model = Some(result.model);
            self.parameter_history.push(result.parameters.values);
            if self.parameter_history.len() > 100 {
                self.parameter_history.remove(0);
            }
        }
        Ok(should_adapt)
    }
    pub fn get_current_model(&self) -> Option<&SystemModel> {
        self.current_model.as_ref()
    }
    pub fn detect_parameter_drift(&self) -> Option<f64> {
        if self.parameter_history.len() < 10 {
            return None;
        }
        let recent_len = 10.min(self.parameter_history.len());
        let recent = &self.parameter_history[self.parameter_history.len() - recent_len..];
        let mut drift_sum = 0.0;
        for i in 1..recent.len() {
            let diff = &recent[i] - &recent[i - 1];
            let diff_norm = diff.iter().map(|&x| x * x).sum::<f64>().sqrt();
            let prev_norm = recent[i - 1].iter().map(|&x| x * x).sum::<f64>().sqrt();
            let change = diff_norm / prev_norm.max(1e-15);
            drift_sum += change;
        }
        Some(drift_sum / (recent.len() - 1) as f64)
    }
}
/// Enhanced system identification result
#[derive(Debug, Clone)]
pub struct EnhancedSysIdResult {
    /// Identified system model
    pub model: SystemModel,
    /// Model parameters with confidence intervals
    pub parameters: ParameterEstimate,
    /// Model validation metrics
    pub validation: ModelValidationMetrics,
    /// Identification method used
    pub method: IdentificationMethod,
    /// Computational diagnostics
    pub diagnostics: ComputationalDiagnostics,
}
/// Computational diagnostics
#[derive(Debug, Clone, Default)]
pub struct ComputationalDiagnostics {
    /// Number of iterations
    pub iterations: usize,
    /// Convergence achieved
    pub converged: bool,
    /// Final cost function value
    pub final_cost: f64,
    /// Condition number of information matrix
    pub condition_number: f64,
    /// Computation time (ms)
    pub computation_time: u128,
}
/// Nonlinear function types
#[derive(Debug, Clone)]
pub enum NonlinearFunction {
    /// Polynomial nonlinearity
    Polynomial(Vec<f64>),
    /// Piecewise linear
    PiecewiseLinear {
        breakpoints: Vec<f64>,
        slopes: Vec<f64>,
    },
    /// Sigmoid function
    Sigmoid { scale: f64, offset: f64 },
    /// Dead zone
    DeadZone { threshold: f64 },
    /// Saturation
    Saturation { lower: f64, upper: f64 },
    /// Custom function
    Custom(String),
}
/// Configuration for enhanced identification
#[derive(Debug, Clone)]
pub struct EnhancedSysIdConfig {
    /// Model structure to identify
    pub model_structure: ModelStructure,
    /// Identification method
    pub method: IdentificationMethod,
    /// Maximum model order
    pub max_order: usize,
    /// Enable order selection
    pub order_selection: bool,
    /// Regularization parameter
    pub regularization: f64,
    /// Forgetting factor for recursive methods
    pub forgetting_factor: f64,
    /// Enable outlier detection
    pub outlier_detection: bool,
    /// Cross-validation folds
    pub cv_folds: Option<usize>,
    /// Use parallel processing
    pub parallel: bool,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Maximum iterations
    pub max_iterations: usize,
}
/// Identification methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IdentificationMethod {
    /// Prediction Error Method
    PEM,
    /// Maximum Likelihood
    MaximumLikelihood,
    /// Subspace identification
    Subspace,
    /// Instrumental Variables
    InstrumentalVariable,
    /// Recursive Least Squares
    RecursiveLeastSquares,
    /// Bayesian identification
    Bayesian,
}
/// Model validation metrics
#[derive(Debug, Clone)]
pub struct ModelValidationMetrics {
    /// Fit percentage on estimation data
    pub fit_percentage: f64,
    /// Cross-validation fit
    pub cv_fit: Option<f64>,
    /// Akaike Information Criterion
    pub aic: f64,
    /// Bayesian Information Criterion
    pub bic: f64,
    /// Final Prediction Error
    pub fpe: f64,
    /// Residual analysis
    pub residual_analysis: ResidualAnalysis,
    /// Stability margin
    pub stability_margin: f64,
}
/// Recursive system identification for online applications
pub struct RecursiveSysId {
    /// Current parameter estimates
    parameters: Array1<f64>,
    /// Covariance matrix
    covariance: Array2<f64>,
    /// Forgetting factor
    lambda: f64,
    /// Buffer for regression vector
    phi_buffer: Vec<f64>,
    /// Model structure
    structure: ModelStructure,
    /// Number of updates
    n_updates: usize,
}
impl RecursiveSysId {
    /// Create new recursive identifier
    pub fn new(initial_params: Array1<f64>, config: &EnhancedSysIdConfig) -> Self {
        let n_params = initial_params.len();
        Self {
            parameters: initial_params,
            covariance: Array2::eye(n_params) * 1000.0,
            lambda: config.forgetting_factor,
            phi_buffer: vec![0.0; n_params],
            structure: config.model_structure,
            n_updates: 0,
        }
    }
    /// Update estimates with new data
    pub fn update(&mut self, input: f64, output: f64) -> SignalResult<f64> {
        self.update_regression_vector(input, output)?;
        let phi = Array1::from_vec(self.phi_buffer.clone());
        let y_pred = self.parameters.dot(&phi);
        let error = output - y_pred;
        let p_phi = self.covariance.dot(&phi);
        let denominator = self.lambda + phi.dot(&p_phi);
        if denominator.abs() > 1e-10 {
            let gain = p_phi / denominator;
            self.parameters = &self.parameters + &gain * error;
            let outer = gain
                .view()
                .insert_axis(Axis(1))
                .dot(&phi.view().insert_axis(Axis(0)));
            self.covariance = (&self.covariance - &outer.dot(&self.covariance)) / self.lambda;
        }
        self.n_updates += 1;
        Ok(error)
    }
    /// Update regression vector
    fn update_regression_vector(&mut self, input: f64, output: f64) -> SignalResult<()> {
        for i in (1..self.phi_buffer.len()).rev() {
            self.phi_buffer[i] = self.phi_buffer[i - 1];
        }
        match self.structure {
            ModelStructure::ARX => {
                let na = self.phi_buffer.len() / 2;
                self.phi_buffer[0] = -output;
                self.phi_buffer[na] = input;
            }
            _ => {
                self.phi_buffer[0] = -output;
            }
        }
        Ok(())
    }
    /// Get current parameter estimates
    pub fn get_parameters(&self) -> &Array1<f64> {
        &self.parameters
    }
    /// Get parameter uncertainties
    pub fn get_uncertainties(&self) -> Array1<f64> {
        self.covariance.diag().map(|x| x.sqrt())
    }
}
/// Parameter estimates with uncertainty
#[derive(Debug, Clone)]
pub struct ParameterEstimate {
    /// Parameter values
    pub values: Array1<f64>,
    /// Covariance matrix
    pub covariance: Array2<f64>,
    /// Standard errors
    pub std_errors: Array1<f64>,
    /// Confidence intervals (95%)
    pub confidence_intervals: Vec<(f64, f64)>,
}
/// System model types
#[derive(Debug, Clone)]
pub enum SystemModel {
    /// Transfer function model
    TransferFunction(TransferFunction),
    /// State-space model
    StateSpace(StateSpace),
    /// ARX model
    ARX {
        a: Array1<f64>,
        b: Array1<f64>,
        delay: usize,
    },
    /// ARMAX model
    ARMAX {
        a: Array1<f64>,
        b: Array1<f64>,
        c: Array1<f64>,
        delay: usize,
    },
    /// Output-Error model
    OE {
        b: Array1<f64>,
        f: Array1<f64>,
        delay: usize,
    },
    /// Box-Jenkins model
    BJ {
        b: Array1<f64>,
        c: Array1<f64>,
        d: Array1<f64>,
        f: Array1<f64>,
        delay: usize,
    },
    /// Hammerstein-Wiener model
    HammersteinWiener {
        linear: Box<SystemModel>,
        input_nonlinearity: NonlinearFunction,
        output_nonlinearity: NonlinearFunction,
    },
}
/// Residual analysis results
#[derive(Debug, Clone)]
pub struct ResidualAnalysis {
    /// Residual autocorrelation
    pub autocorrelation: Array1<f64>,
    /// Cross-correlation with input
    pub cross_correlation: Array1<f64>,
    /// Whiteness test p-value
    pub whiteness_pvalue: f64,
    /// Independence test p-value
    pub independence_pvalue: f64,
    /// Normality test p-value
    pub normality_pvalue: f64,
}
/// Model structure specification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelStructure {
    /// Auto-Regressive with eXogenous input
    ARX,
    /// ARMA with eXogenous input
    ARMAX,
    /// Output-Error
    OE,
    /// Box-Jenkins
    BJ,
    /// State-space
    StateSpace,
    /// Nonlinear ARX
    NARX,
}
/// Model orders for different polynomials
#[derive(Debug, Clone)]
pub struct ModelOrders {
    /// A polynomial order (auto-regressive)
    pub na: usize,
    /// B polynomial order (input)
    pub nb: usize,
    /// C polynomial order (moving average)
    pub nc: usize,
    /// D polynomial order (noise)
    pub nd: usize,
    /// F polynomial order (denominator of input)
    pub nf: usize,
    /// Input delay
    pub nk: usize,
}
