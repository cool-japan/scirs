//! Core data structures and types for enhanced system identification
//!
//! This module contains all the fundamental data structures, enums, and type definitions
//! used throughout the enhanced system identification functionality.

use crate::lti::{StateSpace, TransferFunction};
use scirs2_core::ndarray::{Array1, Array2};

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

/// Parameter estimates with uncertainty quantification
#[derive(Debug, Clone)]
pub struct ParameterEstimate {
    /// Parameter values
    pub values: Array1<f64>,
    /// Covariance matrix of parameter estimates
    pub covariance: Array2<f64>,
    /// Standard errors of parameters
    pub std_errors: Array1<f64>,
    /// Confidence intervals (95%)
    pub confidence_intervals: Vec<(f64, f64)>,
}

/// Comprehensive model validation metrics
#[derive(Debug, Clone)]
pub struct ModelValidationMetrics {
    /// Fit percentage on estimation data
    pub fit_percentage: f64,
    /// Cross-validation fit performance
    pub cv_fit: Option<f64>,
    /// Akaike Information Criterion
    pub aic: f64,
    /// Bayesian Information Criterion
    pub bic: f64,
    /// Final Prediction Error
    pub fpe: f64,
    /// Detailed residual analysis
    pub residual_analysis: ResidualAnalysis,
    /// System stability margin
    pub stability_margin: f64,
}

/// Residual analysis results for model validation
#[derive(Debug, Clone)]
pub struct ResidualAnalysis {
    /// Residual autocorrelation function
    pub autocorrelation: Array1<f64>,
    /// Cross-correlation between residuals and input
    pub cross_correlation: Array1<f64>,
    /// Whiteness test p-value (null hypothesis: residuals are white noise)
    pub whiteness_pvalue: f64,
    /// Independence test p-value (null hypothesis: residuals are independent)
    pub independence_pvalue: f64,
    /// Normality test p-value (null hypothesis: residuals are normally distributed)
    pub normality_pvalue: f64,
}

/// Computational diagnostics and performance metrics
#[derive(Debug, Clone, Default)]
pub struct ComputationalDiagnostics {
    /// Number of optimization iterations performed
    pub iterations: usize,
    /// Whether the algorithm converged to a solution
    pub converged: bool,
    /// Final value of the cost function
    pub final_cost: f64,
    /// Condition number of the information matrix
    pub condition_number: f64,
    /// Total computation time in milliseconds
    pub computation_time: u128,
}

/// Available system identification methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IdentificationMethod {
    /// Prediction Error Method - minimizes prediction error
    PEM,
    /// Maximum Likelihood estimation
    MaximumLikelihood,
    /// Subspace identification methods (e.g., N4SID, MOESP)
    Subspace,
    /// Instrumental Variables method for biased noise scenarios
    InstrumentalVariable,
    /// Recursive Least Squares for online identification
    RecursiveLeastSquares,
    /// Bayesian identification with prior knowledge
    Bayesian,
}

/// Nonlinear function types for Hammerstein-Wiener models
#[derive(Debug, Clone)]
pub enum NonlinearFunction {
    /// Polynomial nonlinearity with coefficients
    Polynomial(Vec<f64>),
    /// Piecewise linear function
    PiecewiseLinear {
        breakpoints: Vec<f64>,
        slopes: Vec<f64>,
    },
    /// Sigmoid function: scale * tanh(x + offset)
    Sigmoid { scale: f64, offset: f64 },
    /// Dead zone nonlinearity
    DeadZone { threshold: f64 },
    /// Saturation nonlinearity
    Saturation { lower: f64, upper: f64 },
    /// Custom nonlinear function (placeholder)
    Custom(String),
}

/// Model structure specification for identification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelStructure {
    /// Auto-Regressive with eXogenous input model
    ARX,
    /// ARMA with eXogenous input model
    ARMAX,
    /// Output-Error model structure
    OE,
    /// Box-Jenkins model structure
    BJ,
    /// State-space representation
    StateSpace,
    /// Nonlinear ARX model
    NARX,
}

/// Model orders for different polynomial components
#[derive(Debug, Clone)]
pub struct ModelOrders {
    /// A polynomial order (auto-regressive part)
    pub na: usize,
    /// B polynomial order (input part)
    pub nb: usize,
    /// C polynomial order (moving average part)
    pub nc: usize,
    /// D polynomial order (noise disturbance)
    pub nd: usize,
    /// F polynomial order (denominator of input transfer function)
    pub nf: usize,
    /// Input delay (number of samples)
    pub nk: usize,
}

impl ModelOrders {
    /// Create default model orders
    pub fn default_arx() -> Self {
        Self {
            na: 2,
            nb: 2,
            nc: 0,
            nd: 0,
            nf: 0,
            nk: 1,
        }
    }

    /// Create default ARMAX model orders
    pub fn default_armax() -> Self {
        Self {
            na: 2,
            nb: 2,
            nc: 2,
            nd: 0,
            nf: 0,
            nk: 1,
        }
    }

    /// Create default Output-Error model orders
    pub fn default_oe() -> Self {
        Self {
            na: 0,
            nb: 2,
            nc: 0,
            nd: 0,
            nf: 2,
            nk: 1,
        }
    }

    /// Create default Box-Jenkins model orders
    pub fn default_bj() -> Self {
        Self {
            na: 0,
            nb: 2,
            nc: 2,
            nd: 2,
            nf: 2,
            nk: 1,
        }
    }
}

/// Configuration for enhanced system identification
#[derive(Debug, Clone)]
pub struct EnhancedSysIdConfig {
    /// Model structure to identify
    pub model_structure: ModelStructure,
    /// Identification method to use
    pub method: IdentificationMethod,
    /// Maximum model order to consider
    pub max_order: usize,
    /// Enable automatic order selection
    pub order_selection: bool,
    /// Regularization parameter for ill-conditioned problems
    pub regularization: f64,
    /// Forgetting factor for recursive methods (0 < λ ≤ 1)
    pub forgetting_factor: f64,
    /// Enable robust outlier detection and removal
    pub outlier_detection: bool,
    /// Number of cross-validation folds for model validation
    pub cv_folds: Option<usize>,
    /// Use parallel processing when available
    pub parallel: bool,
    /// Convergence tolerance for iterative algorithms
    pub tolerance: f64,
    /// Maximum number of iterations
    pub max_iterations: usize,
}

impl Default for EnhancedSysIdConfig {
    fn default() -> Self {
        Self {
            model_structure: ModelStructure::ARX,
            method: IdentificationMethod::PEM,
            max_order: 10,
            order_selection: true,
            regularization: 0.0,
            forgetting_factor: 0.98,
            outlier_detection: false,
            cv_folds: Some(5),
            parallel: true,
            tolerance: 1e-6,
            max_iterations: 100,
        }
    }
}

impl EnhancedSysIdConfig {
    /// Create configuration for ARX identification
    pub fn arx() -> Self {
        Self {
            model_structure: ModelStructure::ARX,
            method: IdentificationMethod::PEM,
            ..Default::default()
        }
    }

    /// Create configuration for ARMAX identification
    pub fn armax() -> Self {
        Self {
            model_structure: ModelStructure::ARMAX,
            method: IdentificationMethod::MaximumLikelihood,
            ..Default::default()
        }
    }

    /// Create configuration for Output-Error identification
    pub fn output_error() -> Self {
        Self {
            model_structure: ModelStructure::OE,
            method: IdentificationMethod::PEM,
            ..Default::default()
        }
    }

    /// Create configuration for state-space identification
    pub fn state_space() -> Self {
        Self {
            model_structure: ModelStructure::StateSpace,
            method: IdentificationMethod::Subspace,
            ..Default::default()
        }
    }

    /// Create configuration for robust identification
    pub fn robust() -> Self {
        Self {
            outlier_detection: true,
            regularization: 1e-3,
            ..Default::default()
        }
    }

    /// Create configuration for recursive identification
    pub fn recursive() -> Self {
        Self {
            method: IdentificationMethod::RecursiveLeastSquares,
            forgetting_factor: 0.95,
            ..Default::default()
        }
    }
}